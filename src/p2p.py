import logging
import httpx
import asyncio
import numpy as np
from typing import List, Dict, Tuple, Sequence, Optional
from sqlmodel import Session, select
from datetime import datetime

from .database import engine, get_topic_collection, get_content_collection
from .models import Peer, Topic
from .config import settings, get_seed_nodes

logger = logging.getLogger(__name__)

async def notify_peer(peer_url: str, action: str, local_topic_id: str, remote_topic_id: str, local_url: str, topic_vector: Optional[Sequence[float]] = None) -> bool:
    """Notifies a peer of an add/remove action so they can mirror it."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            payload = {
                "action": action, # "add" or "remove"
                "peer_url": local_url,
                "local_topic_id": remote_topic_id,
                "remote_topic_id": local_topic_id,
                "topic_vector": list(topic_vector) if topic_vector is not None else None
            }
            response = await client.post(f"{peer_url}/p2p/sync", json=payload)
            response.raise_for_status()
            return True
    except Exception as e:
        logger.warning(f"Failed to notify peer {peer_url} of {action}: {e}")
        return False

def get_peers_for_topic(session: Session, local_topic_id: str) -> Sequence[Peer]:
    return session.exec(select(Peer).where(Peer.local_topic_id == local_topic_id)).all()

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.linalg.norm(v1 - v2))

async def handle_new_peer(
    session: Session, 
    local_topic_id: str, 
    local_topic_vector: Sequence[float], 
    remote_url: str, 
    remote_topic_id: str, 
    remote_topic_vector: Sequence[float]
):
    """
    Implements the Active Swapping Strategy for a newly discovered peer.
    """
    v_local = np.array(local_topic_vector)
    v_remote = np.array(remote_topic_vector)
    sim_score = cosine_similarity(v_local, v_remote)
    
    # Check if this peer is already tracked for this topic
    existing = session.exec(select(Peer).where(
        Peer.url == remote_url, 
        Peer.local_topic_id == local_topic_id
    )).first()
    if existing:
        return

    peers = get_peers_for_topic(session, local_topic_id)
    
    close_limit = int(settings.MAX_PEERS * (settings.CLOSE_PARTITION_PERCENT / 100))
    distant_limit = int(settings.MAX_PEERS * (settings.DISTANT_PARTITION_PERCENT / 100))
    standard_limit = settings.MAX_PEERS - close_limit - distant_limit

    close_peers = [p for p in peers if p.partition_type == "close"]
    distant_peers = [p for p in peers if p.partition_type == "distant"]
    standard_peers = [p for p in peers if p.partition_type == "standard"]

    target_partition = "standard"
    node_to_demote = None

    # Check Close Partition
    if len(close_peers) < close_limit:
        target_partition = "close"
    else:
        # Find least-close node in close partition
        least_close_node = min(close_peers, key=lambda p: p.similarity_score)
        if sim_score > least_close_node.similarity_score:
            target_partition = "close"
            node_to_demote = least_close_node

    # If not close, check Distant Partition
    if target_partition == "standard":
        # Calculate distantness: average distance from a vector to all other known vectors
        all_known_vectors = [v_local]
        for p in peers:
            if p.remote_topic_vector is not None:
                all_known_vectors.append(np.array(p.remote_topic_vector))
                
        def get_distantness(v_target: np.ndarray) -> float:
            if not all_known_vectors:
                return 0.0
            distances = [1.0 - cosine_similarity(v_target, v) for v in all_known_vectors]
            return sum(distances) / len(distances)

        new_node_distantness = get_distantness(v_remote)

        if len(distant_peers) < distant_limit:
            target_partition = "distant"
        else:
            # Find the least-distant node in the distant partition
            least_distant_node = None
            min_distantness = float('inf')
            
            for dp in distant_peers:
                if dp.remote_topic_vector is None:
                    least_distant_node = dp
                    min_distantness = -1.0
                    break
                
                dp_v = np.array(dp.remote_topic_vector)
                dp_distantness = get_distantness(dp_v)
                
                if dp_distantness < min_distantness:
                    min_distantness = dp_distantness
                    least_distant_node = dp
                    
            if new_node_distantness > min_distantness:
                target_partition = "distant"
                node_to_demote = least_distant_node

    # If demoting a node, change its partition to standard
    if node_to_demote:
        node_to_demote.partition_type = "standard"
        session.add(node_to_demote)
        standard_peers.append(node_to_demote)

    # Check Standard Partition capacity
    node_to_remove = None
    if len(standard_peers) >= standard_limit:
        # Drop oldest added standard node
        node_to_remove = min(standard_peers, key=lambda p: p.added_at)
        session.delete(node_to_remove)
        
    new_peer = Peer(
        url=remote_url,
        local_topic_id=local_topic_id,
        remote_topic_id=remote_topic_id,
        remote_topic_vector=list(remote_topic_vector),
        similarity_score=sim_score,
        partition_type=target_partition
    )
    session.add(new_peer)
    session.commit()

    # Async notification to peer that we added them (if it's a new peer, standard notification)
    local_url = f"http://{settings.HOST}:{settings.PORT}"
    asyncio.create_task(notify_peer(remote_url, "add", local_topic_id, remote_topic_id, local_url, local_topic_vector))
    
    if node_to_remove:
        asyncio.create_task(notify_peer(node_to_remove.url, "remove", local_topic_id, node_to_remove.remote_topic_id, local_url))


async def gossip_loop():
    """Background task that periodically contacts peers to discover new vectors."""
    logger.info("Starting P2P Gossip loop")
    while True:
        try:
            with Session(engine) as session:
                topics = session.exec(select(Topic)).all()
                topic_col = get_topic_collection()
                
                local_url = f"http://{settings.HOST}:{settings.PORT}"
                
                for topic in topics:
                    peers = get_peers_for_topic(session, topic.topic_vector_id)
                    
                    # Should we gossip?
                    if len(peers) < settings.MIN_PEERS:
                        # Find targets: seeds + random known peers
                        targets = set(get_seed_nodes())
                        for p in peers:
                            targets.add(p.url)
                            
                        # Discard self
                        targets.discard(local_url)
                        
                        # Get local topic vector
                        topic_data = topic_col.get(ids=[topic.topic_vector_id], include=["embeddings", "metadatas"])
                        embeddings = topic_data.get("embeddings")
                        metadatas = topic_data.get("metadatas")
                        if not embeddings or not metadatas:
                            continue
                            
                        local_vector = [float(x) for x in embeddings[0]]
                        metadata = metadatas[0]
                        
                        for target in targets:
                            try:
                                async with httpx.AsyncClient(timeout=settings.GOSSIP_TIMEOUT) as client:
                                    payload = {
                                        "metadata": metadata,
                                        "topic_vector": local_vector,
                                        "model": settings.EMBEDDING_MODEL,
                                        "requester_url": local_url,
                                        "requester_topic_id": topic.topic_vector_id
                                    }
                                    res = await client.post(f"{target}/p2p/discover", json=payload)
                                    if res.status_code == 200:
                                        data = res.json()
                                        for match in data.get("matches", []):
                                            await handle_new_peer(
                                                session,
                                                topic.topic_vector_id,
                                                local_vector,
                                                match["server_url"],
                                                match["topic_id"],
                                                match["topic_vector"]
                                            )
                            except Exception as e:
                                logger.debug(f"Gossip with {target} failed: {e}")
                                
            await asyncio.sleep(settings.GOSSIP_INTERVAL)
        except Exception as e:
            logger.error(f"Gossip loop error: {e}")
            await asyncio.sleep(settings.GOSSIP_INTERVAL)

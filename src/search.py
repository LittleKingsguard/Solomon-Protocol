import logging
import httpx
import numpy as np
from typing import List, Dict, Any, Optional
from sqlmodel import Session, select

from .database import engine, get_content_collection, get_topic_collection
from .models import Peer
from .indexer import embed_texts
from .config import settings

logger = logging.getLogger(__name__)

async def decentralized_search(query: str, visited_nodes: Optional[List[str]] = None, *, local_only: bool = False) -> Dict[str, Any]:
    """
    Performs a recursive semantic search across the P2P network.
    """
    if visited_nodes is None:
        visited_nodes = []
        
    local_url = f"http://{settings.HOST}:{settings.PORT}"
    if local_url not in visited_nodes:
        visited_nodes.append(local_url)
        
    logger.info(f"Searching for '{query}'. Visited: {visited_nodes}")
    
    # 1. Embed query
    query_vector = embed_texts([query])[0]
    
    # 2. Search local content chunks
    content_col = get_content_collection()
    content_results = content_col.query(
        query_embeddings=[query_vector.tolist()],
        n_results=1,
        include=["documents", "metadatas", "distances"]
    )
    
    best_content_match = None
    best_content_distance = float('inf')
    
    ids = content_results.get("ids")
    distances = content_results.get("distances")
    documents = content_results.get("documents")
    metadatas = content_results.get("metadatas")
    
    if ids and ids[0] and distances and documents and metadatas:
        best_content_distance = distances[0][0]
        best_content_match = {
            "text": documents[0][0],
            "metadata": metadatas[0][0],
            "distance": best_content_distance,
            "source_node": local_url
        }
        
    if local_only:
        if best_content_match:
            return best_content_match
        return {"error": "No content found locally."}
        
    # 3. Search known peer topic vectors directly
    best_peer_match = None
    best_peer_distance = float('inf')
    
    from .p2p import cosine_similarity
    
    with Session(engine) as session:
        peers = session.exec(select(Peer)).all()
        
        # Filter out visited nodes and nodes without vectors
        unvisited_peers = [p for p in peers if p.url not in visited_nodes and p.remote_topic_vector is not None]
        
        for peer in unvisited_peers:
            peer_vector = np.array(peer.remote_topic_vector)
            sim = cosine_similarity(query_vector, peer_vector)
            distance = 1.0 - sim # Convert similarity to a distance metric
            
            if distance < best_peer_distance:
                best_peer_distance = distance
                best_peer_match = peer

    # 4. Resolution Logic
    if best_content_match and (not best_peer_match or best_content_distance <= best_peer_distance):
        # Local content is better or equal
        return best_content_match
    elif best_peer_match:
        # Peer topic is better, forward request
        logger.info(f"Forwarding search to {best_peer_match.url}")
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                payload = {
                    "query": query,
                    "visited_nodes": visited_nodes
                }
                res = await client.post(f"{best_peer_match.url}/p2p/search", json=payload)
                res.raise_for_status()
                return res.json()
        except Exception as e:
            logger.warning(f"Failed to forward search to {best_peer_match.url}: {e}")
            # If peer fails, return our best local content anyway
            if best_content_match:
                return best_content_match
                
    # Fallback
    if best_content_match:
        return best_content_match
        
    return {"error": "No content found in network."}

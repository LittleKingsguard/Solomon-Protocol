from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from sqlmodel import Session, select
import numpy as np

from .database import get_session, get_topic_collection, engine
from .models import Topic, Peer
from .scraper import scrape_pages
from .indexer import index_site_content
from .p2p import handle_new_peer
from .search import decentralized_search

router = APIRouter()

class IndexRequest(BaseModel):
    url: str
    hosting_location: str
    pages_to_index: List[str]

class DiscoverRequest(BaseModel):
    metadata: Dict[str, Any]
    topic_vector: List[float]
    model: str
    requester_url: str
    requester_topic_id: str

class SyncRequest(BaseModel):
    action: str
    peer_url: str
    local_topic_id: str
    remote_topic_id: str
    topic_vector: Optional[List[float]] = None

class SearchRequestPayload(BaseModel):
    query: str
    local_only: bool = False
    visited_nodes: Optional[List[str]] = None

async def bg_index_task(request: IndexRequest):
    """Background task to scrape and index a site."""
    pages_content = await scrape_pages(request.pages_to_index)
    
    metadata = {
        "url": request.url,
        "hosting_location": request.hosting_location
    }
    
    chunk_ids, topic_ids = index_site_content(request.url, metadata, pages_content)
    
    # Save topics to sqlite
    with Session(engine) as session:
        for tid in topic_ids:
            t = Topic(topic_vector_id=tid)
            session.add(t)
        session.commit()

@router.post("/index")
async def index_site(request: IndexRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(bg_index_task, request)
    return {"status": "indexing_started", "url": request.url}

@router.get("/topics")
async def get_topics(session: Session = Depends(get_session)):
    topics = session.exec(select(Topic)).all()
    topic_col = get_topic_collection()
    
    results = []
    for t in topics:
        data = topic_col.get(ids=[t.topic_vector_id], include=["metadatas"])
        if data["ids"]:
            results.append({
                "id": t.topic_vector_id,
                "metadata": data["metadatas"][0] if data["metadatas"] else {}
            })
            
    return {"topics": results}

@router.post("/p2p/discover")
async def p2p_discover(req: DiscoverRequest, session: Session = Depends(get_session)):
    """Receives a topic vector from a peer, finds closest local topics, returns them."""
    topic_col = get_topic_collection()
    
    # Query local ChromaDB for closest topics
    results = topic_col.query(
        query_embeddings=[req.topic_vector],
        n_results=3,
        include=["embeddings", "distances"]
    )
    
    matches = []
    ids = results.get("ids")
    embeddings = results.get("embeddings")
    if ids and ids[0] and embeddings:
        for i, tid in enumerate(ids[0]):
            local_vec = [float(x) for x in embeddings[0][i]]
            
            from .config import settings
            local_url = f"http://{settings.HOST}:{settings.PORT}"
            
            matches.append({
                "topic_id": tid,
                "topic_vector": local_vec,
                "server_url": local_url
            })
            
            # Since they contacted us, we should also track them!
            # Adding them to our peer list for this local topic
            await handle_new_peer(
                session, 
                tid, 
                local_vec, 
                req.requester_url, 
                req.requester_topic_id, 
                req.topic_vector
            )
            
    return {"matches": matches}

@router.post("/p2p/sync")
async def p2p_sync(req: SyncRequest, session: Session = Depends(get_session)):
    """A peer is notifying us that they added/removed us. We mirror the action."""
    if req.action == "add":
        if req.topic_vector:
            topic_col = get_topic_collection()
            data = topic_col.get(ids=[req.local_topic_id], include=["embeddings"])
            
            embeddings = data.get("embeddings")
            if embeddings and len(embeddings) > 0:
                local_vec = [float(x) for x in embeddings[0]]
                await handle_new_peer(
                    session,
                    req.local_topic_id,
                    local_vec,
                    req.peer_url,
                    req.remote_topic_id,
                    req.topic_vector
                )
                return {"status": "ok"}
                
        # Fallback if no topic_vector provided or local topic not found
        existing = session.exec(select(Peer).where(
            Peer.url == req.peer_url, 
            Peer.local_topic_id == req.local_topic_id
        )).first()
        
        if not existing:
            p = Peer(
                url=req.peer_url,
                local_topic_id=req.local_topic_id,
                remote_topic_id=req.remote_topic_id,
                remote_topic_vector=req.topic_vector,
                similarity_score=0.0,
                partition_type="standard"
            )
            session.add(p)
            session.commit()
            
    elif req.action == "remove":
        p = session.exec(select(Peer).where(
            Peer.url == req.peer_url, 
            Peer.local_topic_id == req.local_topic_id
        )).first()
        if p:
            session.delete(p)
            session.commit()
            
    return {"status": "ok"}

@router.post("/search")
@router.post("/p2p/search")
async def execute_search(req: SearchRequestPayload):
    """Executes a recursive search query."""
    visited_nodes = req.visited_nodes if req.visited_nodes is not None else []
    result = await decentralized_search(req.query, visited_nodes, local_only=req.local_only)
    return result

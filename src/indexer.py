import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import uuid
from .config import settings
from .database import get_content_collection, get_topic_collection

logger = logging.getLogger(__name__)

# Load model globally to avoid reloading
try:
    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
    model = SentenceTransformer(settings.EMBEDDING_MODEL)
except Exception as e:
    logger.error(f"Failed to load embedding model. Ensure you have sufficient resources. Error: {e}")
    model = None

def chunk_text(text: str) -> List[str]:
    """Splits text into overlapping chunks based on settings."""
    chunk_size = settings.CHUNK_SIZE
    overlap = settings.CHUNK_OVERLAP
    words = text.split()
    chunks = []
    
    if not words:
        return chunks
        
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    """Generates embeddings for a list of strings."""
    if model is None:
        raise RuntimeError("Embedding model is not loaded.")
    return np.asarray(model.encode(texts))

def cluster_embeddings(embeddings: np.ndarray) -> List[int]:
    """
    Clusters embeddings using DBSCAN to automatically determine the number of topics.
    Returns a list of cluster labels (-1 means noise/outlier).
    """
    # Require a cluster to have at least 5% of the total embeddings to avoid
    # creating tiny clusters from single-article tangents, with a floor of 2.
    total_embeddings = len(embeddings)
    dynamic_min_samples = max(2, int(total_embeddings * 0.05))
    
    # Using cosine distance, DBSCAN eps is a distance threshold
    # eps=0.3 means vectors with cosine distance < 0.3 are grouped
    clustering = DBSCAN(eps=0.3, min_samples=dynamic_min_samples, metric='cosine')
    labels = clustering.fit_predict(embeddings)
    return labels.tolist()

def calculate_centroids(embeddings: np.ndarray, labels: List[int]) -> Dict[int, np.ndarray]:
    """Calculates the average vector (centroid) for each cluster."""
    centroids = {}
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            continue # Skip noise
        cluster_points = embeddings[np.array(labels) == label]
        centroid = np.mean(cluster_points, axis=0)
        # Normalize the centroid for cosine similarity
        centroid = centroid / np.linalg.norm(centroid)
        centroids[label] = centroid
    return centroids

def index_site_content(url: str, metadata: dict, pages_content: Dict[str, Tuple[str, List[str]]]) -> Tuple[List[str], List[str]]:
    """
    Takes the scraped content of a site, chunks it, embeds it, clusters it into topics,
    and saves both the chunks and the topic centroids to ChromaDB.
    Returns the generated topic IDs.
    """
    all_chunks = []
    chunk_metadata = []
    
    for page_url, (text, images) in pages_content.items():
        if text:
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                meta = metadata.copy()
                meta.update({"source": page_url, "chunk_index": i, "site_url": url, "type": "text"})
                chunk_metadata.append(meta)
                
        for i, img_md in enumerate(images):
            all_chunks.append(img_md)
            meta = metadata.copy()
            meta.update({"source": page_url, "chunk_index": i, "site_url": url, "type": "image"})
            chunk_metadata.append(meta)
            
    if not all_chunks:
        logger.warning(f"No content to index for {url}")
        return [], []
        
    embeddings = embed_texts(all_chunks)
    labels = cluster_embeddings(embeddings)
    centroids = calculate_centroids(embeddings, labels)
    
    content_col = get_content_collection()
    topic_col = get_topic_collection()
    
    chunk_ids = [str(uuid.uuid4()) for _ in all_chunks]
    
    # Store chunks
    content_col.add(
        ids=chunk_ids,
        embeddings=embeddings.tolist(),
        documents=all_chunks,
        metadatas=chunk_metadata
    )
    
    # Store topics
    topic_ids = []
    topic_embeddings = []
    topic_metadatas = []
    
    for label, centroid in centroids.items():
        topic_id = str(uuid.uuid4())
        topic_ids.append(topic_id)
        topic_embeddings.append(centroid.tolist())
        topic_metadatas.append({
            "site_url": url,
            "cluster_id": int(label),
            "num_chunks": int(np.sum(np.array(labels) == label))
        })
        
    if topic_ids:
        topic_col.add(
            ids=topic_ids,
            embeddings=topic_embeddings,
            metadatas=topic_metadatas
        )
        
    return chunk_ids, topic_ids

import os
from sqlmodel import SQLModel, Session, create_engine
import chromadb
from chromadb.config import Settings as ChromaSettings

sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# We will use two collections: one for raw content chunks, one for topic centroids
content_collection = chroma_client.get_or_create_collection(
    name="content_chunks",
    metadata={"hnsw:space": "cosine"}
)

topic_collection = chroma_client.get_or_create_collection(
    name="topics",
    metadata={"hnsw:space": "cosine"}
)

def get_content_collection():
    return content_collection

def get_topic_collection():
    return topic_collection

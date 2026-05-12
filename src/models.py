from typing import Optional
from sqlmodel import Field, SQLModel
from datetime import datetime

class Topic(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    topic_vector_id: str = Field(index=True, unique=True) # ID in ChromaDB
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # Could add metadata here like the number of chunks that formed this topic

from sqlalchemy import Column, JSON

class Peer(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    url: str = Field(index=True)
    local_topic_id: str = Field(index=True) # Corresponding ChromaDB ID of the local topic
    remote_topic_id: str = Field() # Corresponding ChromaDB ID of the remote topic
    remote_topic_vector: Optional[list[float]] = Field(default=None, sa_column=Column(JSON))
    similarity_score: float = Field()
    partition_type: str = Field() # "close", "distant", "standard"
    added_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="active") # "active", "unreachable"

    class Config:
        # url + local_topic_id must be unique to prevent duplicates per topic
        unique_together = ("url", "local_topic_id")

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_PEERS: int = 100
    MIN_PEERS: int = 90
    CLOSE_PARTITION_PERCENT: int = 20
    DISTANT_PARTITION_PERCENT: int = 20
    SEED_NODES: str = ""
    EMBEDDING_MODEL: str = "Qwen/Qwen3-VL-Embedding-2B"
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    GOSSIP_TIMEOUT: float = 10.0
    GOSSIP_INTERVAL: int = 60

    class Config:
        env_file = ".env"

settings = Settings()

def get_seed_nodes():
    if not settings.SEED_NODES:
        return []
    # Splits comma-separated string, strips whitespace, and filters out empty entries
    return [node.strip() for node in settings.SEED_NODES.split(",") if node.strip()]

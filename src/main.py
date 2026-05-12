from contextlib import asynccontextmanager
from fastapi import FastAPI
import asyncio
import logging

from .database import create_db_and_tables
from .api import router
from .p2p import gossip_loop
from .config import settings

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    # Start background gossip loop
    gossip_task = asyncio.create_task(gossip_loop())
    yield
    # Clean up gossip loop on shutdown
    gossip_task.cancel()

app = FastAPI(title="Solomon Protocol Server", lifespan=lifespan)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)

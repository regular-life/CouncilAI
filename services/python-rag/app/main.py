"""PadhAI-Dost Python RAG Service — FastAPI entrypoint."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import ingest, retrieve, retrieve_all
from app.models import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Python RAG Service...")
    settings = get_settings()

    from app.embedding.transformer import TransformerEmbeddings

    logger.info(f"Preloading embedding model: {settings.embedding_model}")
    TransformerEmbeddings(settings.embedding_model)
    logger.info("Embedding model loaded successfully")

    yield
    logger.info("Shutting down Python RAG Service")


app = FastAPI(
    title="PadhAI-Dost RAG Service",
    description="Document intelligence API with adaptive OCR, layout-aware chunking, and semantic retrieval.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(retrieve.router, tags=["Retrieval"])
app.include_router(retrieve_all.router, tags=["Retrieval"])


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )

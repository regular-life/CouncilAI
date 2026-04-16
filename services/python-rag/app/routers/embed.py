"""
Embed Router — POST /embed

Provides an isolated embedding generation endpoint for raw text.
Used by the Go Backend for fast Semantic Caching.
"""

import logging
from fastapi import APIRouter, HTTPException

from app.models import EmbedRequest, EmbedResponse
from app.config import get_settings
from app.embedding.transformer import TransformerEmbeddings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """
    Generate a 384-dimensional vector embedding for the input text.
    Bypasses the vector store completely.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        settings = get_settings()
        # The model is preloaded into the class-level cache in main.py
        embeddings = TransformerEmbeddings(settings.embedding_model)
        
        # embed_query takes a single string and returns a list of floats
        vector = embeddings.embed_query(request.text)

        return EmbedResponse(embedding=vector)

    except Exception as e:
        logger.exception(f"Embedding failed for text: {request.text[:50]}...")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

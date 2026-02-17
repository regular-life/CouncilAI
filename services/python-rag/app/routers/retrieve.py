"""
Retrieve Router — POST /retrieve

Handles query embedding and top-K chunk retrieval from ChromaDB.
"""

import logging
from fastapi import APIRouter, HTTPException

from app.models import RetrieveRequest, RetrieveResponse
from app.retrieval.chroma_store import ChromaStore

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_chunks(request: RetrieveRequest):
    """
    Retrieve relevant chunks for a given question.

    Returns top-K chunks ordered by relevance score,
    with metadata (chunk type, page number, etc.).
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        store = ChromaStore()
        chunks = store.retrieve(
            query=request.question,
            doc_id=request.doc_id,
            top_k=request.top_k,
        )

        return RetrieveResponse(
            chunks=chunks,
            question=request.question,
            doc_id=request.doc_id,
        )

    except Exception as e:
        logger.exception(f"Retrieval failed for question: {request.question[:50]}...")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

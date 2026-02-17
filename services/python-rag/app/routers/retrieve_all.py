"""Retrieve-All Router — returns all stored chunks for a document."""

import logging
from fastapi import APIRouter, HTTPException

from app.models import RetrieveAllRequest, RetrieveAllResponse, Chunk, ChunkType
from app.retrieval.chroma_store import ChromaStore

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/retrieve-all", response_model=RetrieveAllResponse)
async def retrieve_all_chunks(request: RetrieveAllRequest):
    """Retrieve all chunks for a document, ordered by page and chunk index."""
    try:
        store = ChromaStore()
        collection_name = store._collection_name(request.doc_id)

        from langchain_community.vectorstores import Chroma

        db = Chroma(
            persist_directory=store.persist_directory,
            embedding_function=store.embeddings,
            collection_name=collection_name,
        )

        results = db.get(include=["documents", "metadatas"])

        if not results or not results.get("documents"):
            raise HTTPException(
                status_code=404,
                detail=f"Document {request.doc_id} not found or has no content",
            )

        chunks = []
        for i, (doc, meta) in enumerate(
            zip(results["documents"], results.get("metadatas", [{}] * len(results["documents"])))
        ):
            meta = meta or {}
            chunks.append(
                Chunk(
                    content=doc,
                    chunk_type=ChunkType(meta.get("chunk_type", "paragraph")),
                    page_number=meta.get("page_number", 0),
                    chunk_index=meta.get("chunk_index", i),
                    metadata=meta,
                )
            )

        chunks.sort(key=lambda c: (c.page_number, c.chunk_index))

        logger.info(f"Retrieved all {len(chunks)} chunks for doc_id={request.doc_id}")

        return RetrieveAllResponse(
            chunks=chunks,
            doc_id=request.doc_id,
            total_chunks=len(chunks),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Retrieve-all failed for doc_id={request.doc_id}")
        raise HTTPException(status_code=500, detail=f"Retrieve-all failed: {str(e)}")

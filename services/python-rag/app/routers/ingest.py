"""
Ingest Router — POST /ingest

Handles document upload, inspection, OCR routing,
layout-aware chunking, embedding, and storage in ChromaDB.
"""

import hashlib
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.models import IngestResponse
from app.inspection.inspector import inspect_document
from app.ocr.router import route_ocr
from app.chunking.layout_chunker import chunk_document
from app.retrieval.chroma_store import ChromaStore

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    doc_id: str | None = None,
):
    """
    Ingest a document through the full pipeline:
    1. Inspect document metadata
    2. Route to appropriate OCR backend
    3. Chunk with layout awareness
    4. Embed and store in ChromaDB
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Read file bytes
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Generate doc_id if not provided
    if not doc_id:
        file_hash = hashlib.md5(file_bytes[:4096]).hexdigest()[:12]
        doc_id = f"{file.filename}_{file_hash}"

    logger.info(f"Ingesting document: {file.filename} (doc_id={doc_id})")

    try:
        # Step 1: Inspect
        metadata = inspect_document(file_bytes, file.filename)
        logger.info(
            f"Inspection result: text_layer={metadata.has_text_layer}, "
            f"scanned={metadata.is_scanned}, tables={metadata.has_tables}, "
            f"multicolumn={metadata.is_multicolumn}"
        )

        # Step 2: OCR Route
        ocr_result = route_ocr(file_bytes, file.filename, metadata)
        logger.info(
            f"OCR method: {ocr_result.ocr_method}, blocks: {len(ocr_result.blocks)}"
        )

        if not ocr_result.blocks:
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted from the document",
            )

        # Step 3: Layout-aware chunking
        chunks = chunk_document(ocr_result)
        logger.info(f"Chunks created: {len(chunks)}")

        # Step 4: Embed and store
        store = ChromaStore()
        chunk_count = store.ingest(chunks, doc_id)

        return IngestResponse(
            doc_id=doc_id,
            chunk_count=chunk_count,
            metadata=metadata,
            message=f"Successfully ingested {chunk_count} chunks using {ocr_result.ocr_method} OCR",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Ingestion failed for {file.filename}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

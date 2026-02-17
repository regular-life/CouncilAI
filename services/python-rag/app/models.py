"""
Pydantic models for the Python RAG service.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ChunkType(str, Enum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CAPTION = "caption"
    HEADING = "heading"
    LIST = "list"
    UNKNOWN = "unknown"


class DocumentMetadata(BaseModel):
    has_text_layer: bool = False
    is_scanned: bool = False
    has_tables: bool = False
    is_multicolumn: bool = False
    page_count: int = 0
    file_type: str = ""
    file_name: str = ""


class Chunk(BaseModel):
    content: str
    chunk_type: ChunkType = ChunkType.PARAGRAPH
    page_number: int = 0
    chunk_index: int = 0
    metadata: dict = Field(default_factory=dict)


class OCRBlock(BaseModel):
    content: str
    block_type: ChunkType = ChunkType.PARAGRAPH
    page_number: int = 0
    bbox: Optional[list[float]] = None
    confidence: float = 1.0


class OCRResult(BaseModel):
    blocks: list[OCRBlock] = Field(default_factory=list)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    ocr_method: str = ""


class IngestRequest(BaseModel):
    doc_id: Optional[str] = None


class IngestResponse(BaseModel):
    doc_id: str
    chunk_count: int
    metadata: DocumentMetadata
    message: str = "Document ingested successfully"


class RetrieveRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None
    top_k: int = 5


class RetrieveResponse(BaseModel):
    chunks: list[Chunk]
    question: str
    doc_id: Optional[str] = None


class RetrieveAllRequest(BaseModel):
    doc_id: str


class RetrieveAllResponse(BaseModel):
    chunks: list[Chunk]
    doc_id: str
    total_chunks: int


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    service: str = "python-rag"

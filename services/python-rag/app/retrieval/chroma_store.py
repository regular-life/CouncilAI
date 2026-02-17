"""
ChromaDB Vector Store Operations.

Handles document ingestion into ChromaDB and retrieval of
relevant chunks with metadata.
"""

import logging
import hashlib
from typing import Optional

from langchain_community.vectorstores import Chroma

from app.models import Chunk, ChunkType
from app.embedding.transformer import TransformerEmbeddings
from app.config import get_settings

logger = logging.getLogger(__name__)


class ChromaStore:
    """
    ChromaDB-backed vector store for document chunks.
    Supports per-document collections and metadata filtering.
    """

    def __init__(self, persist_directory: Optional[str] = None):
        settings = get_settings()
        self.persist_directory = persist_directory or settings.chroma_db_path
        self.embeddings = TransformerEmbeddings(settings.embedding_model)

    def ingest(self, chunks: list[Chunk], doc_id: str) -> int:
        """
        Ingest chunks into ChromaDB with metadata.

        Args:
            chunks: List of Chunk objects to store.
            doc_id: Document identifier for collection naming.

        Returns:
            Number of chunks ingested.
        """
        if not chunks:
            logger.warning("No chunks to ingest")
            return 0

        texts = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "doc_id": doc_id,
                "chunk_type": chunk.chunk_type.value,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                **{k: str(v) for k, v in chunk.metadata.items()},
            }
            for chunk in chunks
        ]

        # Generate unique IDs for each chunk
        ids = [
            f"{doc_id}_{chunk.chunk_index}_{hashlib.md5(chunk.content[:100].encode()).hexdigest()[:8]}"
            for chunk in chunks
        ]

        db = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            ids=ids,
            persist_directory=self.persist_directory,
            collection_name=self._collection_name(doc_id),
        )
        db.persist()

        logger.info(f"Ingested {len(chunks)} chunks for doc_id={doc_id}")
        return len(chunks)

    def retrieve(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[Chunk]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: The search query.
            doc_id: Optional document ID to filter by.
            top_k: Number of results to return.

        Returns:
            List of Chunk objects ordered by relevance.
        """
        collection_name = self._collection_name(doc_id) if doc_id else "default"

        try:
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name,
            )
        except Exception as e:
            logger.error(f"Failed to open ChromaDB collection '{collection_name}': {e}")
            return []

        results = db.similarity_search_with_relevance_scores(query, k=top_k)

        chunks = []
        for doc, score in results:
            meta = doc.metadata or {}
            chunk = Chunk(
                content=doc.page_content,
                chunk_type=ChunkType(meta.get("chunk_type", "paragraph")),
                page_number=int(meta.get("page_number", 0)),
                chunk_index=int(meta.get("chunk_index", 0)),
                metadata={**meta, "relevance_score": score},
            )
            chunks.append(chunk)

        logger.info(f"Retrieved {len(chunks)} chunks for query (doc_id={doc_id})")
        return chunks

    def delete_collection(self, doc_id: str) -> bool:
        """Delete a document's collection from ChromaDB."""
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self.persist_directory)
            collection_name = self._collection_name(doc_id)
            client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    @staticmethod
    def _collection_name(doc_id: Optional[str]) -> str:
        """Generate a valid ChromaDB collection name from doc_id."""
        if not doc_id:
            return "default"
        # ChromaDB collection names must be 3-63 chars, alphanumeric + underscores
        safe_name = "".join(c if c.isalnum() else "_" for c in doc_id)
        return f"doc_{safe_name}"[:63]

    def get_document_text(self, doc_id: str) -> str:
        """
        Retrieve all text from a document's collection.
        Used for samjha_do and pucho features.
        """
        collection_name = self._collection_name(doc_id)
        try:
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name,
            )
            # Get all documents from the collection
            results = db.get()
            if results and results.get("documents"):
                return "\n\n".join(results["documents"])
            return ""
        except Exception as e:
            logger.error(f"Failed to get document text for {doc_id}: {e}")
            return ""

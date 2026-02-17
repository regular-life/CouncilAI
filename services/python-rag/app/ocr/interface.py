"""
Pluggable OCR interface — all OCR backends must implement this.
"""

from abc import ABC, abstractmethod

from app.models import OCRResult


class OCRBackend(ABC):
    """Abstract base class for OCR backends."""

    @abstractmethod
    def process(self, file_bytes: bytes, filename: str) -> OCRResult:
        """
        Process a document and return structured OCR output.

        Args:
            file_bytes: Raw file content bytes.
            filename: Original filename.

        Returns:
            OCRResult with structured blocks and metadata.
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return the name of this OCR backend."""
        ...

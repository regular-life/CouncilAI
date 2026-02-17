"""
Document Inspection Layer.

Analyzes uploaded documents to determine their characteristics:
- Does it have an embedded text layer?
- Is it a scanned image?
- Does it contain tables?
- Is it a multi-column layout?

This metadata drives the adaptive OCR routing decisions.
"""

import io
import logging
from pathlib import Path

from PyPDF2 import PdfReader
from PIL import Image

from app.models import DocumentMetadata

logger = logging.getLogger(__name__)


def inspect_document(file_bytes: bytes, filename: str) -> DocumentMetadata:
    """
    Inspect a document and produce structured metadata.

    Args:
        file_bytes: Raw file content bytes.
        filename: Original filename for type detection.

    Returns:
        DocumentMetadata with inspection results.
    """
    ext = Path(filename).suffix.lower()
    metadata = DocumentMetadata(file_name=filename, file_type=ext)

    if ext == ".pdf":
        metadata = _inspect_pdf(file_bytes, metadata)
    elif ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        metadata = _inspect_image(file_bytes, metadata)
    else:
        logger.warning(f"Unsupported file type: {ext}")

    return metadata


def _inspect_pdf(file_bytes: bytes, metadata: DocumentMetadata) -> DocumentMetadata:
    """Inspect a PDF document for text layer, tables, and layout."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        metadata.page_count = len(reader.pages)

        total_text_chars = 0
        total_pages_with_text = 0
        table_indicators = 0

        for page in reader.pages:
            text = page.extract_text() or ""
            char_count = len(text.strip())
            total_text_chars += char_count

            if char_count > 50:
                total_pages_with_text += 1

            # Heuristic: lines with multiple tab/pipe separators suggest tables
            for line in text.split("\n"):
                if line.count("|") >= 2 or line.count("\t") >= 2:
                    table_indicators += 1

        # Determine text layer presence
        if total_pages_with_text > 0:
            text_coverage = total_pages_with_text / metadata.page_count
            metadata.has_text_layer = text_coverage > 0.5
            metadata.is_scanned = text_coverage < 0.3
        else:
            metadata.has_text_layer = False
            metadata.is_scanned = True

        # Table detection heuristic
        metadata.has_tables = table_indicators >= 3

        # Multi-column heuristic: check if pages have wide spacing patterns
        # This is a simple heuristic — layout-aware OCR handles the real detection
        if total_text_chars > 0 and total_pages_with_text > 0:
            avg_chars_per_page = total_text_chars / total_pages_with_text
            # Multi-column docs tend to have many short lines
            sample_page = reader.pages[0]
            sample_text = sample_page.extract_text() or ""
            lines = [l for l in sample_text.split("\n") if l.strip()]
            if lines:
                avg_line_length = sum(len(l) for l in lines) / len(lines)
                # Short avg line length with high char count → likely multi-column
                metadata.is_multicolumn = avg_line_length < 60 and avg_chars_per_page > 500

    except Exception as e:
        logger.error(f"PDF inspection failed: {e}")
        metadata.is_scanned = True  # Fallback: treat as scanned

    return metadata


def _inspect_image(file_bytes: bytes, metadata: DocumentMetadata) -> DocumentMetadata:
    """Inspect an image file — always treated as scanned."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        metadata.page_count = 1
        metadata.is_scanned = True
        metadata.has_text_layer = False

        # Table detection heuristic for images: check aspect ratio
        # Wide images with structured content might be tables
        width, height = img.size
        if width > height * 1.5:
            metadata.has_tables = True

    except Exception as e:
        logger.error(f"Image inspection failed: {e}")
        metadata.is_scanned = True

    return metadata

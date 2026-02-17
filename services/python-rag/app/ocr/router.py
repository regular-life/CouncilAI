"""
Adaptive OCR Router.

Selects the appropriate OCR backend based on document inspection metadata:
- Text layer exists → direct text extraction (skip OCR)
- Tables detected → layout-aware OCR (pdfplumber)
- Scanned/image → Tesseract OCR
"""

import io
import logging
from pathlib import Path

from PyPDF2 import PdfReader

from app.models import DocumentMetadata, OCRResult, OCRBlock, ChunkType
from app.ocr.interface import OCRBackend
from app.ocr.tesseract import TesseractOCR
from app.ocr.layout_aware import LayoutAwareOCR

logger = logging.getLogger(__name__)


class DirectTextExtractor(OCRBackend):
    """
    'OCR' backend that simply extracts embedded text from PDFs.
    Used when the document already has a text layer — no OCR needed.
    """

    def name(self) -> str:
        return "direct_text"

    def process(self, file_bytes: bytes, filename: str) -> OCRResult:
        """Extract text directly from PDF text layer."""
        blocks: list[OCRBlock] = []
        ext = Path(filename).suffix.lower()

        if ext != ".pdf":
            logger.warning("DirectTextExtractor only supports PDFs")
            return OCRResult(
                blocks=blocks,
                metadata=DocumentMetadata(file_name=filename, file_type=ext),
                ocr_method=self.name(),
            )

        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    # Split into paragraphs
                    paragraphs = text.split("\n\n")
                    for para in paragraphs:
                        if para.strip():
                            blocks.append(
                                OCRBlock(
                                    content=para.strip(),
                                    block_type=ChunkType.PARAGRAPH,
                                    page_number=page_num,
                                    confidence=1.0,
                                )
                            )
        except Exception as e:
            logger.error(f"Direct text extraction failed: {e}")

        return OCRResult(
            blocks=blocks,
            metadata=DocumentMetadata(
                file_name=filename,
                file_type=ext,
                page_count=len(blocks),
            ),
            ocr_method=self.name(),
        )


def route_ocr(file_bytes: bytes, filename: str, metadata: DocumentMetadata) -> OCRResult:
    """
    Route to the appropriate OCR backend based on document metadata.

    Decision logic:
        if text layer exists:
            skip OCR → direct text extraction
        elif tables detected:
            use layout-aware OCR
        else:
            use Tesseract OCR
    """
    backend: OCRBackend

    if metadata.has_text_layer and not metadata.is_scanned:
        if metadata.has_tables:
            logger.info(f"Routing {filename} → layout-aware OCR (text + tables)")
            backend = LayoutAwareOCR()
        else:
            logger.info(f"Routing {filename} → direct text extraction (text layer found)")
            backend = DirectTextExtractor()
    elif metadata.has_tables:
        logger.info(f"Routing {filename} → layout-aware OCR (tables detected)")
        backend = LayoutAwareOCR()
    else:
        logger.info(f"Routing {filename} → Tesseract OCR (scanned/image)")
        backend = TesseractOCR()

    result = backend.process(file_bytes, filename)
    result.ocr_method = backend.name()

    return result

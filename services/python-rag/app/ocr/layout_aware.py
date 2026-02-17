"""
Layout-aware OCR backend.

Uses pdfplumber for structured extraction of tables, paragraphs,
and layout information from PDF documents.
"""

import io
import logging
from pathlib import Path

from app.models import OCRBlock, OCRResult, DocumentMetadata, ChunkType
from app.ocr.interface import OCRBackend

logger = logging.getLogger(__name__)


class LayoutAwareOCR(OCRBackend):
    """
    Layout-aware OCR backend using pdfplumber.
    Extracts tables as structured blocks and paragraphs separately.
    """

    def name(self) -> str:
        return "layout_aware"

    def process(self, file_bytes: bytes, filename: str) -> OCRResult:
        """Extract structured content from PDFs using pdfplumber."""
        ext = Path(filename).suffix.lower()

        if ext != ".pdf":
            logger.warning(
                f"LayoutAwareOCR is optimized for PDFs, got {ext}. "
                "Falling back to basic extraction."
            )
            return self._fallback_extraction(file_bytes, filename, ext)

        return self._process_pdf(file_bytes, filename)

    def _process_pdf(self, file_bytes: bytes, filename: str) -> OCRResult:
        """Process a PDF with layout-aware extraction."""
        import pdfplumber

        blocks: list[OCRBlock] = []
        page_count = 0

        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                page_count = len(pdf.pages)

                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract tables first
                    tables = page.extract_tables()
                    table_bboxes = []

                    if tables:
                        for table_obj in page.find_tables():
                            table_bboxes.append(table_obj.bbox)

                        for table in tables:
                            table_text = self._format_table(table)
                            if table_text.strip():
                                blocks.append(
                                    OCRBlock(
                                        content=table_text,
                                        block_type=ChunkType.TABLE,
                                        page_number=page_num,
                                        confidence=0.9,
                                    )
                                )

                    # Extract text outside tables
                    text = page.extract_text() or ""
                    if text.strip():
                        # Split into paragraphs by double newlines
                        paragraphs = self._split_paragraphs(text)
                        for para in paragraphs:
                            if para.strip():
                                # Try to classify the paragraph
                                block_type = self._classify_block(para)
                                blocks.append(
                                    OCRBlock(
                                        content=para.strip(),
                                        block_type=block_type,
                                        page_number=page_num,
                                        confidence=0.95,
                                    )
                                )

        except Exception as e:
            logger.error(f"Layout-aware OCR failed: {e}")

        return OCRResult(
            blocks=blocks,
            metadata=DocumentMetadata(
                file_name=filename,
                file_type=".pdf",
                page_count=page_count,
                has_tables=any(b.block_type == ChunkType.TABLE for b in blocks),
            ),
            ocr_method=self.name(),
        )

    def _format_table(self, table: list[list]) -> str:
        """Format a table as a markdown-style string."""
        if not table:
            return ""

        rows = []
        for row in table:
            cells = [str(cell).strip() if cell else "" for cell in row]
            rows.append("| " + " | ".join(cells) + " |")

        # Add header separator after first row
        if len(rows) > 1:
            header = rows[0]
            separator = "| " + " | ".join(["---"] * len(table[0])) + " |"
            rows.insert(1, separator)

        return "\n".join(rows)

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs using heuristics."""
        # Split on double newlines
        paragraphs = text.split("\n\n")

        # If no double newlines, try to split on single newlines with
        # heuristic: a new paragraph starts with a capital letter after
        # a line ending with a period
        if len(paragraphs) <= 1:
            lines = text.split("\n")
            result: list[str] = []
            current: list[str] = []

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    if current:
                        result.append(" ".join(current))
                        current = []
                    continue

                if (
                    current
                    and current[-1].rstrip().endswith(".")
                    and stripped[0].isupper()
                ):
                    result.append(" ".join(current))
                    current = [stripped]
                else:
                    current.append(stripped)

            if current:
                result.append(" ".join(current))

            return result

        return paragraphs

    def _classify_block(self, text: str) -> ChunkType:
        """Classify a text block by type using heuristics."""
        stripped = text.strip()

        # Heading: short, no period at end, often capitalized
        if len(stripped) < 100 and not stripped.endswith(".") and stripped[0].isupper():
            words = stripped.split()
            if len(words) <= 10:
                return ChunkType.HEADING

        # Caption: starts with "Figure", "Table", "Fig.", etc.
        caption_prefixes = ("figure", "fig.", "fig ", "table", "chart", "diagram")
        if stripped.lower().startswith(caption_prefixes):
            return ChunkType.CAPTION

        # List: starts with bullet or number
        if stripped.startswith(("•", "-", "*", "1.", "2.", "3.")):
            return ChunkType.LIST

        return ChunkType.PARAGRAPH

    def _fallback_extraction(
        self, file_bytes: bytes, filename: str, ext: str
    ) -> OCRResult:
        """Fallback for non-PDF files."""
        from app.ocr.tesseract import TesseractOCR

        return TesseractOCR().process(file_bytes, filename)

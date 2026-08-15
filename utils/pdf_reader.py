"""
pdf_reader.py

Thin wrapper around pypdf for extracting text from an uploaded PDF.
Kept separate from the rest of the app so the extraction strategy (and its
limitations) is easy to find and explain: this does NOT do OCR, so a
scanned/image-only PDF will come back empty — the app checks for that and
tells the user, instead of silently proceeding with no context.
"""

from io import BytesIO
from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages_text = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(text)

    return "\n\n".join(pages_text)

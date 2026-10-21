
from pathlib import Path

from pypdf import PdfReader


def load_document(file_path: str) -> list[dict]:
    """Load TXT or PDF text with source metadata."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    records = []

    if path.suffix.lower() == ".txt":
        text = path.read_text(encoding="utf-8").strip()

        if text:
            records.append({
                "text": text,
                "source": path.name,
                "page": None,
            })

    elif path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()

            if page_text:
                records.append({
                    "text": page_text,
                    "source": path.name,
                    "page": page_number,
                })

    else:
        raise ValueError("Only TXT and PDF files are supported.")

    if not records:
        raise ValueError("Document contains no extractable text.")

    return records

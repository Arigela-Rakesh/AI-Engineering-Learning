from pathlib import Path

import pymupdf


def load_pdf(file_path: str) -> str:
    """
    Extract text from a PDF document.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text from the PDF.
    """

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    extracted_text = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text()

            if page_text.strip():
                extracted_text.append(
                    f"\n--- Page {page_number} ---\n{page_text}"
                )

    return "\n".join(extracted_text)
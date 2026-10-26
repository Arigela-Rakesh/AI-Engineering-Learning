
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[str]:
    """Split document text at numbered policy headings."""

    if not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    text = text.replace("\r\n", "\n").strip()

    # Recognize headings such as:
    # 1. ANNUAL LEAVE POLICY
    # 8. INTERNET AND EXPENSE REIMBURSEMENT
    # 18. FREQUENTLY ASKED QUESTIONS
    heading_pattern = re.compile(
        r"(?m)^\s*\d+\.\s+[A-Z][A-Z0-9\s&,()/\-]+$"
    )

    matches = list(heading_pattern.finditer(text))

    if not matches:
        sections = [text]
    else:
        sections = []

        # Preserve any text before the first numbered heading.
        prefix = text[:matches[0].start()].strip()
        if prefix:
            sections.append(prefix)

        # Keep each heading with the content until the next heading.
        for i, match in enumerate(matches):
            start = match.start()

            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                end = len(text)

            section = text[start:end].strip()

            if section:
                sections.append(section)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []

    for section in sections:
        chunks.extend(splitter.split_text(section))

    return chunks

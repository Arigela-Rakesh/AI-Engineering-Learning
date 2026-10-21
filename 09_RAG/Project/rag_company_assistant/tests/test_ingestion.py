from pathlib import Path

from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_text


def main():
    file_path = Path("data/documents/company_policy.txt")

    text = load_document(str(file_path))
    chunks = chunk_text(text, chunk_size=300, overlap=50)

    print("Extracted text:")
    print(text)

    print("\nNumber of chunks:", len(chunks))

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk)


if __name__ == "__main__":
    main()

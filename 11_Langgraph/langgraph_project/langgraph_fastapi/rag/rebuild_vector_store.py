
from pathlib import Path

from rag.document_loader import load_pdf
from rag.text_splitter import split_text
from rag.vector_store import build_faiss_index, save_vector_store


PDF_PATH = Path("sample.pdf")
VECTOR_STORE_PATH = Path("rag/vector_store")


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF file not found: {PDF_PATH.resolve()}"
        )

    print("Loading handbook PDF...")

    document_text = load_pdf(str(PDF_PATH))

    print("Splitting document with updated splitter...")

    chunks = split_text(
        document_text,
        chunk_size=500,
        chunk_overlap=100,
    )

    if not chunks:
        raise ValueError("No chunks were generated from the PDF.")

    print("Building a fresh FAISS index...")

    index, chunks = build_faiss_index(chunks)

    save_vector_store(
        index,
        chunks,
        str(VECTOR_STORE_PATH),
    )

    print("\nFresh vector store created successfully.")
    print(f"Total chunks: {len(chunks)}")
    print(f"FAISS vectors: {index.ntotal}")
    print(f"Saved to: {VECTOR_STORE_PATH.resolve()}")

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} ---")
        print(chunk)


if __name__ == "__main__":
    main()


from pathlib import Path

from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_text
from app.retrieval.vector_store import VectorStore


DOCUMENT_PATH = Path("data/documents/company_policy.txt")
QUESTION = "What is the internet reimbursement limit?"


def build_store():
    records = load_document(str(DOCUMENT_PATH))
    documents = []

    for record in records:
        chunks = chunk_text(
            record["text"],
            chunk_size=25,
            overlap=5,
        )

        for chunk_id, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "source": record["source"],
                "page": record["page"],
                "chunk_id": chunk_id,
            })

    store = VectorStore()
    store.add_documents(documents)
    return store


def main():
    store = build_store()

    for top_k in [1, 2, 3, 5]:
        results = store.search(QUESTION, top_k=top_k)

        print(f"\n{'=' * 50}")
        print(f"top_k = {top_k}")
        print(f"Number of retrieved chunks: {len(results)}")

        for rank, result in enumerate(results, start=1):
            print(
                f"\nRank: {rank} | "
                f"Chunk: {result['chunk_id']} | "
                f"Similarity: {result['score']:.3f}"
            )
            print(result["text"])


if __name__ == "__main__":
    main()

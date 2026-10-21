from pathlib import Path

from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_text
from app.retrieval.vector_store import VectorStore


def main():
    document_path = Path("data/documents/company_policy.txt")

    document_text = load_document(str(document_path))
    chunks = chunk_text(
        document_text,
        chunk_size=60,
        overlap=10,
    )

    vector_store = VectorStore()
    vector_store.add_documents(chunks)

    questions = [
        "How many annual leave days do employees receive?",
        "What is the internet reimbursement limit?",
        "How many days can employees work from home?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 60)

        results = vector_store.search(question, top_k=2)

        for result in results:
            print(f"Similarity: {result['score']}")
            print(f"Retrieved text: {result['chunk']}\n")


if __name__ == "__main__":
    main()

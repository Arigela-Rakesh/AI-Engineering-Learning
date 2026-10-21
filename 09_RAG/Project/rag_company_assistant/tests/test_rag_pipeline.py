from pathlib import Path

from app.rag.pipeline import RAGPipeline
from app.rag.generator import generate_answer


def main():
    document_path = Path("data/documents/company_policy.txt")

    pipeline = RAGPipeline(
        document_path=str(document_path),
        chunk_size=60,
        overlap=10,
    )

    print("Ingesting document...")
    pipeline.ingest()

    print(f"Total chunks indexed: {len(pipeline.chunks)}")

    questions = [
        "How many annual leave days do employees receive?",
        "What is the internet reimbursement limit?",
        "How many days can employees work from home?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")

        results = pipeline.ask(question, top_k=2)

        for rank, result in enumerate(results, start=1):
            print(f"\nResult {rank}")
            print(f"Similarity: {result['score']}")
            print(f"Text: {result['chunk']}")


if __name__ == "__main__":
    main()

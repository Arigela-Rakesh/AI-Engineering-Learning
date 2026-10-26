
from rag.rag_pipeline import initialize_vector_store
from rag.vector_store import search_documents


def main():
    index, chunks = initialize_vector_store()

    questions = [
        "What is the internet reimbursement amount?",
        "How much money is provided monthly for internet expenses?",
        "What is the capital of India?",
    ]

    for question in questions:
        print("\n" + "=" * 70)
        print(f"Question: {question}")
        print("=" * 70)

        results = search_documents(
            query=question,
            index=index,
            chunks=chunks,
            top_k=5,
        )

        for rank, result in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print(f"Chunk index: {result['chunk_index']}")
            print(f"Similarity score: {result['score']:.4f}")
            print(f"Chunk text:\n{result['chunk']}")


if __name__ == "__main__":
    main()

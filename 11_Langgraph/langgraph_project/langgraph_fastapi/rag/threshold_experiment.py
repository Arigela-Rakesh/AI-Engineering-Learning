
from rag.rag_pipeline import initialize_vector_store
from rag.vector_store import search_documents


def main():
    index, chunks = initialize_vector_store()

    questions = [
        "What is the internet reimbursement amount?",
        "How much money is provided monthly for internet expenses?",
        "What is the capital of India?",
    ]

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]

    for question in questions:
        print("\n" + "=" * 75)
        print(f"Question: {question}")
        print("=" * 75)

        results = search_documents(
            query=question,
            index=index,
            chunks=chunks,
            top_k=5,
        )

        for threshold in thresholds:
            filtered_results = []

            for result in results:
                if result["score"] >= threshold:
                    filtered_results.append(result)

            print(f"\nThreshold: {threshold:.2f}")
            print(f"Matching chunks: {len(filtered_results)}")

            if len(filtered_results) == 0:
                print("No sufficiently similar context found.")
            else:
                for result in filtered_results:
                    print(
                        f"Chunk {result['chunk_index']} | "
                        f"Score: {result['score']:.4f}"
                    )


if __name__ == "__main__":
    main()

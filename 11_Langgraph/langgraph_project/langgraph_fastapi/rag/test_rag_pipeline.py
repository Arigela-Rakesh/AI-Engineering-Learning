
from rag.rag_pipeline import (
    initialize_vector_store,
    ask_question,
)


def main():
    index, chunks = initialize_vector_store()

    questions = [
        "What is the internet reimbursement amount?",
        "What is the capital of India?",
    ]

    for question in questions:
        result = ask_question(
            question=question,
            index=index,
            chunks=chunks,
        )

        print("\n" + "=" * 70)
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Source count: {result['source_count']}")

        for source in result["sources"]:
            print(f"\nSource: {source['title']}")
            print(f"Chunk index: {source['chunk_index']}")
            print(
                f"Similarity: {source['similarity_score']:.4f}"
            )
            print(f"Evidence:\n{source['text']}")


if __name__ == "__main__":
    main()

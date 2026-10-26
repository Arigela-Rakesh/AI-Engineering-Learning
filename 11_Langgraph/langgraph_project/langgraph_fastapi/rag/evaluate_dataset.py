
from rag.rag_pipeline import (
    initialize_vector_store,
    retrieve_context,
)

from rag.evaluation_dataset import EVALUATION_DATASET


def evaluate_retrieval():

    print("Initializing vector store...")

    index, chunks = initialize_vector_store()

    print("Vector store initialized.")

    total_tests = len(EVALUATION_DATASET)

    total_passed = 0

    category_results = {}

    for test_case in EVALUATION_DATASET:

        question = test_case["question"]

        category = test_case["category"]

        should_retrieve = test_case["should_retrieve"]

        print("\n" + "=" * 70)

        print(f"Test ID: {test_case['id']}")

        print(f"Category: {category}")

        print(f"Question: {question}")

        # Retrieve context from FAISS.
        context, retrieved_results = retrieve_context(
            question=question,
            index=index,
            chunks=chunks,
        )

        retrieved_count = len(retrieved_results)

        # Determine expected retrieval behavior.
        if should_retrieve:
            passed = retrieved_count > 0
        else:
            passed = retrieved_count == 0

        if passed:
            total_passed += 1

        # Store category metrics.
        if category not in category_results:
            category_results[category] = {
                "total": 0,
                "passed": 0,
            }

        category_results[category]["total"] += 1

        if passed:
            category_results[category]["passed"] += 1

        print(f"Retrieved chunks: {retrieved_count}")

        print(
            f"Retrieval result: "
            f"{'PASS' if passed else 'FAIL'}"
        )

        # Display retrieved chunk information.
        for result in retrieved_results:
           chunk_index = result.get("chunk_index")
           score = result.get("score")
           print(f"\nChunk index: {chunk_index}")
           print(f"Similarity score: {score}")

           if chunk_index is not None:
               chunk_text = chunks[chunk_index]

               print(f"Excerpt: {chunk_text[:300]}")

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    pass_rate = (
        total_passed / total_tests
    ) * 100

    print("\n" + "=" * 70)

    print("RETRIEVAL EVALUATION SUMMARY")

    print("=" * 70)

    print(f"Total tests: {total_tests}")

    print(f"Passed: {total_passed}")

    print(f"Failed: {total_tests - total_passed}")

    print(f"Overall pass rate: {pass_rate:.2f}%")

    print("\nCategory-wise results:")

    for category, result in category_results.items():

        category_pass_rate = (
            result["passed"] / result["total"]
        ) * 100

        print(
            f"{category}: "
            f"{result['passed']}/{result['total']} "
            f"({category_pass_rate:.2f}%)"
        )

    print("=" * 70)


if __name__ == "__main__":
    evaluate_retrieval()

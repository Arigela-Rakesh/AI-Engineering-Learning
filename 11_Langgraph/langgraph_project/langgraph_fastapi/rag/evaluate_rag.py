
from rag.rag_pipeline import (
    initialize_vector_store,
    retrieve_context,
)


# ---------------------------------------
# 1. Evaluation Test Cases
# ---------------------------------------

EVALUATION_DATASET = [
    {
        "question": "How many annual leave days are provided?",
        "expected_terms": [
            "annual leave",
            "20 days",
        ],
        "should_retrieve": True,
    },
    {
        "question": "What is the internet reimbursement amount?",
        "expected_terms": [
            "internet",
            "1500",
        ],
        "should_retrieve": True,
    },
    {
        "question": "How many maternity leave days are provided?",
        "expected_terms": [
            "maternity",
            "HR",
        ],
        "should_retrieve": True,
    },
    {
        "question": "What is the capital city of India?",
        "expected_terms": [],
        "should_retrieve": False,
    },
]


# ---------------------------------------
# 2. Initialize the Vector Store
# ---------------------------------------

def initialize_evaluation():

    print("Initializing vector store...")

    index, chunks = initialize_vector_store()

    print("Vector store initialized.")

    return index, chunks


# ---------------------------------------
# 3. Evaluate Retrieval
# ---------------------------------------

def evaluate_retrieval(
    index,
    chunks,
):
    """
    Evaluate whether relevant information is retrieved.

    This is a simple keyword-based evaluation.
    It is useful as a starting point, but it is not
    a complete semantic relevance evaluator.
    """

    total_tests = 0
    passed_tests = 0

    for test_case in EVALUATION_DATASET:

        total_tests += 1

        question = test_case["question"]

        expected_terms = test_case["expected_terms"]

        should_retrieve = test_case["should_retrieve"]

        print("\n" + "=" * 60)

        print("Question:", question)

        # Retrieve context using the existing RAG pipeline.
        context, results = retrieve_context(
            question=question,
            index=index,
            chunks=chunks,
        )

        # Determine whether any relevant chunks were retrieved.
        retrieved_context = context.lower()

        matched_terms = []

        for term in expected_terms:

            if term.lower() in retrieved_context:

                matched_terms.append(term)

        if should_retrieve:

            # For handbook questions, we expect relevant context.
            if len(results) > 0 and len(matched_terms) > 0:

                passed = True

            else:

                passed = False

        else:

            # For out-of-scope questions, we expect no context.
            if len(results) == 0:

                passed = True

            else:

                passed = False

        if passed:

            passed_tests += 1

        # Display evaluation results.
        if passed:

            print("Status: PASS")

        else:

            print("Status: FAIL")

        print("Retrieved chunks:", len(results))

        print("Matched expected terms:", matched_terms)

        print("Expected terms:", expected_terms)

        # Display retrieved chunk details.
        for result in results:

            print(
                "Chunk:",
                result["chunk_index"],
                "| Score:",
                round(result["score"], 4),
            )

    # ---------------------------------------
    # 4. Evaluation Summary
    # ---------------------------------------

    print("\n" + "=" * 60)

    print("RAG RETRIEVAL EVALUATION SUMMARY")

    print("=" * 60)

    print("Total tests:", total_tests)

    print("Passed tests:", passed_tests)

    print("Failed tests:", total_tests - passed_tests)

    if total_tests > 0:

        accuracy = (
            passed_tests / total_tests
        ) * 100

        print(
            "Evaluation accuracy:",
            round(accuracy, 2),
            "%",
        )


# ---------------------------------------
# 5. Main Function
# ---------------------------------------

def main():

    index, chunks = initialize_evaluation()

    evaluate_retrieval(
        index=index,
        chunks=chunks,
    )


if __name__ == "__main__":

    main()

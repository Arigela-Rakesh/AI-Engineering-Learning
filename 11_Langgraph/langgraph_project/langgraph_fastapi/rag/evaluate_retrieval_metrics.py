
from typing import List, Set

from rag.rag_pipeline import (
    initialize_vector_store,
    retrieve_context,
)

from rag.retrieval_metrics_dataset import (
    EVALUATION_DATASET,
)


# --------------------------------------------------
# 1. Precision@K
# --------------------------------------------------

def precision_at_k(
    retrieved_chunks: List[int],
    relevant_chunks: Set[int],
    k: int,
) -> float:

    if k <= 0:
        return 0.0

    top_k = retrieved_chunks[:k]

    if len(top_k) == 0:
        return 0.0

    relevant_count = 0

    for chunk_index in top_k:

        if chunk_index in relevant_chunks:
            relevant_count += 1

    return relevant_count / len(top_k)


# --------------------------------------------------
# 2. Recall@K
# --------------------------------------------------

def recall_at_k(
    retrieved_chunks: List[int],
    relevant_chunks: Set[int],
    k: int,
) -> float:

    if k <= 0 or len(relevant_chunks) == 0:
        return 0.0

    top_k = retrieved_chunks[:k]

    relevant_count = 0

    for chunk_index in top_k:

        if chunk_index in relevant_chunks:
            relevant_count += 1

    return relevant_count / len(relevant_chunks)


# --------------------------------------------------
# 3. Reciprocal Rank
# --------------------------------------------------

def reciprocal_rank(
    retrieved_chunks: List[int],
    relevant_chunks: Set[int],
) -> float:

    rank = 1

    for chunk_index in retrieved_chunks:

        if chunk_index in relevant_chunks:
            return 1 / rank

        rank += 1

    return 0.0


# --------------------------------------------------
# 4. Evaluate actual FAISS retrieval
# --------------------------------------------------

def evaluate_retrieval_metrics():

    print("Initializing vector store...")

    index, chunks = initialize_vector_store()

    print("Vector store initialized.")

    k = 3

    total_precision = 0.0
    total_recall = 0.0
    total_mrr = 0.0

    total_tests = len(EVALUATION_DATASET)

    for test_case in EVALUATION_DATASET:

        question = test_case["question"]

        relevant_chunks = set(
            test_case["relevant_chunk_indices"]
        )

        print("\n" + "=" * 70)

        print(f"Test ID: {test_case['id']}")

        print(f"Question: {question}")

        print(
            f"Expected relevant chunks: "
            f"{relevant_chunks}"
        )

        # Retrieve actual chunks from FAISS.
        context, retrieved_results = retrieve_context(
            question=question,
            index=index,
            chunks=chunks,
        )

        retrieved_chunks = []

        for result in retrieved_results:

            chunk_index = result.get("chunk_index")

            if chunk_index is not None:
                retrieved_chunks.append(chunk_index)

        # Calculate metrics.
        precision = precision_at_k(
            retrieved_chunks,
            relevant_chunks,
            k,
        )

        recall = recall_at_k(
            retrieved_chunks,
            relevant_chunks,
            k,
        )

        mrr = reciprocal_rank(
            retrieved_chunks,
            relevant_chunks,
        )

        total_precision += precision
        total_recall += recall
        total_mrr += mrr

        print(
            f"Retrieved chunks: "
            f"{retrieved_chunks}"
        )

        print(
            f"Precision@{k}: "
            f"{precision:.2f}"
        )

        print(
            f"Recall@{k}: "
            f"{recall:.2f}"
        )

        print(
            f"Reciprocal Rank: "
            f"{mrr:.2f}"
        )

        # Display retrieved chunk details.
        for result in retrieved_results:

            chunk_index = result.get("chunk_index")

            score = result.get("score")

            print(
                f"\nChunk index: {chunk_index}"
            )

            print(
                f"Similarity score: {score}"
            )

            if chunk_index is not None:

                print(
                    f"Relevant: "
                    f"{chunk_index in relevant_chunks}"
                )

                print(
                    f"Excerpt: "
                    f"{chunks[chunk_index][:200]}"
                )

    # --------------------------------------------------
    # 5. Calculate average metrics
    # --------------------------------------------------

    if total_tests == 0:
        print("No evaluation test cases found.")
        return

    average_precision = (
        total_precision / total_tests
    )

    average_recall = (
        total_recall / total_tests
    )

    average_mrr = (
        total_mrr / total_tests
    )

    print("\n" + "=" * 70)

    print("RETRIEVAL METRICS SUMMARY")

    print("=" * 70)

    print(
        f"Total test cases: "
        f"{total_tests}"
    )

    print(
        f"Average Precision@{k}: "
        f"{average_precision:.2f}"
    )

    print(
        f"Average Recall@{k}: "
        f"{average_recall:.2f}"
    )

    print(
        f"Mean Reciprocal Rank (MRR): "
        f"{average_mrr:.2f}"
    )

    print("=" * 70)


if __name__ == "__main__":
    evaluate_retrieval_metrics()

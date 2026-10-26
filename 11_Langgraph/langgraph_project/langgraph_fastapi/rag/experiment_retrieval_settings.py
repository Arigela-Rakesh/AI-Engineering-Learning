
from typing import List, Set

from rag.rag_pipeline import (
    initialize_vector_store,
    retrieve_context,
)

from rag.retrieval_metrics_dataset import (
    EVALUATION_DATASET,
)


# --------------------------------------------------
# 1. Metric functions
# --------------------------------------------------

def precision_at_k(
    retrieved_chunks: List[int],
    relevant_chunks: Set[int],
    k: int,
) -> float:

    if k <= 0:
        return 0.0

    top_k = retrieved_chunks[:k]

    if not top_k:
        return 0.0

    relevant_count = 0

    for chunk_index in top_k:

        if chunk_index in relevant_chunks:
            relevant_count += 1

    return relevant_count / len(top_k)


def recall_at_k(
    retrieved_chunks: List[int],
    relevant_chunks: Set[int],
    k: int,
) -> float:

    if k <= 0 or not relevant_chunks:
        return 0.0

    top_k = retrieved_chunks[:k]

    relevant_count = 0

    for chunk_index in top_k:

        if chunk_index in relevant_chunks:
            relevant_count += 1

    return relevant_count / len(relevant_chunks)


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
# 2. Experiment configurations
# --------------------------------------------------

EXPERIMENTS = [
    {
        "name": "A",
        "top_k": 3,
        "min_score": 0.40,
    },
    {
        "name": "B",
        "top_k": 3,
        "min_score": 0.45,
    },
    {
        "name": "C",
        "top_k": 3,
        "min_score": 0.50,
    },
    {
        "name": "D",
        "top_k": 2,
        "min_score": 0.40,
    },
    {
        "name": "E",
        "top_k": 5,
        "min_score": 0.40,
    },
]


# --------------------------------------------------
# 3. Run experiments
# --------------------------------------------------

def run_experiments():

    print("Initializing vector store...")

    index, chunks = initialize_vector_store()

    print("Vector store initialized.")

    total_tests = len(EVALUATION_DATASET)

    if total_tests == 0:
        print("No evaluation questions found.")
        return

    experiment_results = []

    for experiment in EXPERIMENTS:

        experiment_name = experiment["name"]

        top_k = experiment["top_k"]

        min_score = experiment["min_score"]

        total_precision = 0.0
        total_recall = 0.0
        total_mrr = 0.0

        print("\n" + "=" * 70)

        print(f"Experiment: {experiment_name}")

        print(f"TOP_K: {top_k}")

        print(f"MIN_SIMILARITY_SCORE: {min_score}")

        print("=" * 70)

        for test_case in EVALUATION_DATASET:

            question = test_case["question"]

            relevant_chunks = set(
                test_case["relevant_chunk_indices"]
            )

            # Retrieve using this experiment's settings.
            context, retrieved_results = retrieve_context(
                question=question,
                index=index,
                chunks=chunks,
                top_k=top_k,
                min_score=min_score,
            )

            retrieved_chunks = []

            for result in retrieved_results:

                chunk_index = result.get("chunk_index")

                if chunk_index is not None:
                    retrieved_chunks.append(chunk_index)

            precision = precision_at_k(
                retrieved_chunks,
                relevant_chunks,
                top_k,
            )

            recall = recall_at_k(
                retrieved_chunks,
                relevant_chunks,
                top_k,
            )

            mrr = reciprocal_rank(
                retrieved_chunks,
                relevant_chunks,
            )

            total_precision += precision
            total_recall += recall
            total_mrr += mrr

            print(
                f"\n{test_case['id']}: "
                f"Retrieved={retrieved_chunks}, "
                f"Precision={precision:.2f}, "
                f"Recall={recall:.2f}, "
                f"RR={mrr:.2f}"
            )

        # Calculate average metrics.
        average_precision = (
            total_precision / total_tests
        )

        average_recall = (
            total_recall / total_tests
        )

        average_mrr = (
            total_mrr / total_tests
        )

        result = {
            "name": experiment_name,
            "top_k": top_k,
            "min_score": min_score,
            "precision": average_precision,
            "recall": average_recall,
            "mrr": average_mrr,
        }

        experiment_results.append(result)

        print("\nExperiment Summary:")

        print(
            f"Average Precision@{top_k}: "
            f"{average_precision:.2f}"
        )

        print(
            f"Average Recall@{top_k}: "
            f"{average_recall:.2f}"
        )

        print(
            f"Mean Reciprocal Rank: "
            f"{average_mrr:.2f}"
        )

    # --------------------------------------------------
    # 4. Compare all experiments
    # --------------------------------------------------

    print("\n" + "=" * 90)

    print("FINAL EXPERIMENT COMPARISON")

    print("=" * 90)

    print(
        f"{'Experiment':<12}"
        f"{'TOP_K':<10}"
        f"{'Threshold':<12}"
        f"{'Precision':<14}"
        f"{'Recall':<12}"
        f"{'MRR':<10}"
    )

    print("-" * 90)

    for result in experiment_results:

        print(
            f"{result['name']:<12}"
            f"{result['top_k']:<10}"
            f"{result['min_score']:<12.2f}"
            f"{result['precision']:<14.2f}"
            f"{result['recall']:<12.2f}"
            f"{result['mrr']:<10.2f}"
        )

    print("=" * 90)


if __name__ == "__main__":
    run_experiments()

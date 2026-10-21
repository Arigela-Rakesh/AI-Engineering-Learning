
def precision_at_k(
    retrieved_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    top_k = retrieved_ids[:k]
    relevant_count = sum(
        1 for chunk_id in top_k if chunk_id in relevant_ids
    )

    # Precision@K uses K as the denominator.
    return relevant_count / k


def recall_at_k(
    retrieved_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty.")

    top_k = retrieved_ids[:k]
    relevant_count = sum(
        1 for chunk_id in top_k if chunk_id in relevant_ids
    )

    return relevant_count / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: list[int],
    relevant_ids: set[int],
) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0

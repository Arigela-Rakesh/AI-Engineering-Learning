
import pytest

from app.retrieval.evaluation import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k():
    retrieved = [10, 20, 30]
    relevant = {10, 30}

    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(2 / 3)


def test_recall_at_k():
    retrieved = [10, 20, 30]
    relevant = {10, 30}

    assert recall_at_k(retrieved, relevant, 3) == 1.0


def test_reciprocal_rank_first_result():
    retrieved = [10, 20, 30]
    relevant = {10, 30}

    assert reciprocal_rank(retrieved, relevant) == 1.0


def test_reciprocal_rank_second_result():
    retrieved = [20, 30, 10]
    relevant = {10, 30}

    assert reciprocal_rank(retrieved, relevant) == 0.5


def test_reciprocal_rank_no_relevant_result():
    retrieved = [20, 40, 50]
    relevant = {10, 30}

    assert reciprocal_rank(retrieved, relevant) == 0.0


def test_precision_rejects_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k([10, 20], {10}, 0)


def test_recall_rejects_empty_relevant_set():
    with pytest.raises(ValueError):
        recall_at_k([10, 20], set(), 2)

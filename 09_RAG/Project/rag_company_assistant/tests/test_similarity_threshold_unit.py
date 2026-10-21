
import pytest

from app.retrieval.vector_store import VectorStore


@pytest.fixture
def store():
    vector_store = VectorStore()

    vector_store.add_documents([
        {
            "text": "Employees receive 20 days of annual leave.",
            "source": "test.txt",
            "page": None,
            "chunk_id": 0,
        },
        {
            "text": "The internet reimbursement limit is 1500 rupees.",
            "source": "test.txt",
            "page": None,
            "chunk_id": 1,
        },
    ])

    return vector_store


def test_search_returns_results_without_threshold(store):
    results = store.search(
        "internet reimbursement limit",
        top_k=2,
    )

    assert len(results) == 2


def test_threshold_filters_low_scores(store):
    results = store.search(
        "internet reimbursement limit",
        top_k=2,
        min_score=0.99,
    )

    for result in results:
        assert result["score"] >= 0.99


def test_threshold_rejects_invalid_value(store):
    with pytest.raises(ValueError):
        store.search(
            "internet reimbursement limit",
            min_score=1.5,
        )


def test_search_rejects_zero_top_k(store):
    with pytest.raises(ValueError):
        store.search(
            "internet reimbursement limit",
            top_k=0,
        )


from unittest.mock import Mock, patch

from app.rag.pipeline import RAGPipeline


def test_answer_returns_fallback_when_no_chunks_match():
    pipeline = RAGPipeline(document_path="unused.txt")
    pipeline.vector_store = Mock()
    pipeline.vector_store.search.return_value = []

    with patch("app.rag.generator.generate_answer") as mock_generator:
        result = pipeline.answer(
            question="What is the company's maternity leave policy?",
            top_k=2,
            min_score=0.70,
        )

    assert result["sources"] == []
    assert result["retrieved_chunks"] == []
    assert "couldn't find sufficiently relevant information" in result["answer"]
    mock_generator.assert_not_called()

from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from app.main import app


def test_root_endpoint():
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Company Document Q&A API"


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "pipeline_ready": True,
    }


def test_ask_success():
    fake_pipeline = Mock()
    fake_pipeline.chunks = [{"text": "Mock company policy"}]
    fake_pipeline.answer.return_value = {
        "question": "How many annual leave days do employees receive?",
        "answer": "Employees receive 20 days of annual leave per calendar year.",
        "sources": [
            {
                "source": "company_policy.txt",
                "page": None,
                "chunk_id": 0,
            }
        ],
        "retrieved_chunks": [],
    }

    with patch("app.main.RAGPipeline", return_value=fake_pipeline):
        with TestClient(app) as client:
            response = client.post(
                "/ask",
                json={
                    "question": "How many annual leave days do employees receive?",
                    "top_k": 2,
                },
            )

    assert response.status_code == 200
    assert "20 days" in response.json()["answer"]
    assert len(response.json()["sources"]) == 1
    fake_pipeline.answer.assert_called_once_with(
        question="How many annual leave days do employees receive?",
        top_k=2,
        min_score=None,
    )
    fake_pipeline.ingest.assert_called_once()



def test_ask_rejects_missing_question():
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={"top_k": 2},
        )

    assert response.status_code == 422


def test_ask_rejects_invalid_top_k():
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "question": "What is the leave policy?",
                "top_k": 10,
            },
        )

    assert response.status_code == 422


def test_ask_rejects_empty_question():
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "question": "",
                "top_k": 2,
            },
        )

    assert response.status_code == 422



def test_ask_rejects_invalid_min_score():
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "question": "What is the reimbursement limit?",
                "top_k": 2,
                "min_score": 1.5,
            },
        )

    assert response.status_code == 422


def test_ask_accepts_valid_min_score():
    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "question": "What is the reimbursement limit?",
                "top_k": 2,
                "min_score": 0.3,
            },
        )

    assert response.status_code == 200

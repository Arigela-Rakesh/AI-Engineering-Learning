from fastapi.testclient import TestClient
from app import main

client = TestClient(main.app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == ("AI Text Assistant API is running")


def test_sentiment_success(monkeypatch):
    def fake_predict_sentiment(text):
        return {
            "text": text,
            "sentiment": "POSITIVE",
            "confidence": 0.95,
        }

    monkeypatch.setattr(
        main,
        "predict_sentiment",
        fake_predict_sentiment,
    )

    response = client.post(
        "/sentiment",
        json={"text": "I love this product"},
    )

    assert response.status_code == 200
    assert response.json()["sentiment"] == "POSITIVE"
    assert response.json()["confidence"] == 0.95


def test_sentiment_missing_text():
    response = client.post("/sentiment", json={})

    assert response.status_code == 422


def test_sentiment_empty_text():
    response = client.post(
        "/sentiment",
        json={"text": ""},
    )

    assert response.status_code == 422


def test_generation_success(monkeypatch):
    def fake_generate_text(
        prompt,
        max_new_tokens=80,
        temperature=0.8,
    ):
        return {
            "prompt": prompt,
            "generated_text": "help people solve problems.",
            "settings": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3,
            },
        }

    monkeypatch.setattr(
        main,
        "generate_text",
        fake_generate_text,
    )

    response = client.post(
        "/generate",
        json={
            "prompt": "AI can",
            "max_new_tokens": 50,
            "temperature": 0.7,
        },
    )

    assert response.status_code == 200
    assert response.json()["prompt"] == "AI can"
    assert "generated_text" in response.json()
    assert response.json()["settings"]["max_new_tokens"] == 50


def test_generation_invalid_temperature():
    response = client.post(
        "/generate",
        json={
            "prompt": "AI can",
            "temperature": 0,
        },
    )

    assert response.status_code == 422


def test_generation_excessive_tokens():
    response = client.post(
        "/generate",
        json={
            "prompt": "AI can",
            "max_new_tokens": 500,
        },
    )

    assert response.status_code == 422

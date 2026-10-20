import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from app.config import SENTIMENT_MODEL_NAME

tokenizer = None
model = None


def load_sentiment_model():
    global tokenizer, model

    if model is not None and tokenizer is not None:
        return

    loaded_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    loaded_model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL_NAME
    )
    loaded_model.eval()

    tokenizer = loaded_tokenizer
    model = loaded_model


def predict_sentiment(text: str) -> dict:
    if model is None or tokenizer is None:
        raise RuntimeError("Sentiment model is not loaded")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)

    predicted_class = probabilities.argmax(dim=-1).item()
    confidence = probabilities[0, predicted_class].item()
    label = model.config.id2label[predicted_class]

    return {
        "text": text,
        "sentiment": label,
        "confidence": round(confidence, 4),
    }

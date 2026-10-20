import os

SENTIMENT_MODEL_NAME = os.getenv(
    "SENTIMENT_MODEL_NAME",
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
)

GENERATION_MODEL_NAME = os.getenv(
    "GENERATION_MODEL_NAME",
    "Qwen/Qwen2.5-0.5B-Instruct",
)

MAX_SENTIMENT_TEXT_LENGTH = int(os.getenv("MAX_SENTIMENT_TEXT_LENGTH", "5000"))

MAX_GENERATION_PROMPT_LENGTH = int(os.getenv("MAX_GENERATION_PROMPT_LENGTH", "2000"))

DEFAULT_MAX_NEW_TOKENS = int(os.getenv("DEFAULT_MAX_NEW_TOKENS", "80"))

MAX_NEW_TOKENS_LIMIT = int(os.getenv("MAX_NEW_TOKENS_LIMIT", "200"))

DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.8"))

FRONTEND_ORIGIN = os.getenv(
    "FRONTEND_ORIGIN",
    "http://localhost:5173",
)

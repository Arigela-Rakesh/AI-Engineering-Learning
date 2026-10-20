from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    FRONTEND_ORIGIN,
    MAX_GENERATION_PROMPT_LENGTH,
    MAX_NEW_TOKENS_LIMIT,
    MAX_SENTIMENT_TEXT_LENGTH,
)
from app.model import load_sentiment_model, predict_sentiment
from app.generation import load_generation_model, generate_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading AI models")

    try:
        load_sentiment_model()
        load_generation_model()
        logger.info("AI models loaded successfully")
        yield
    finally:
        logger.info("Application lifespan ended")


app = FastAPI(
    title="AI Text Assistant",
    description="Sentiment analysis and text generation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class SentimentRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=MAX_SENTIMENT_TEXT_LENGTH,
    )


class GenerationRequest(BaseModel):
    prompt: str = Field(
        min_length=1,
        max_length=MAX_GENERATION_PROMPT_LENGTH,
    )
    max_new_tokens: int = Field(
        default=DEFAULT_MAX_NEW_TOKENS,
        ge=1,
        le=MAX_NEW_TOKENS_LIMIT,
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        gt=0,
        le=2,
    )


@app.get("/")
def home():
    return {
        "message": "AI Text Assistant API is running",
        "endpoints": [
            "/sentiment",
            "/generate",
            "/health",
            "/docs",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sentiment")
def analyze_sentiment(request: SentimentRequest):
    try:
        logger.info("Received sentiment request")
        return predict_sentiment(request.text)
    except Exception:
        logger.exception("Sentiment inference failed")
        raise HTTPException(
            status_code=500,
            detail="Sentiment analysis failed. Please try again later.",
        ) from None


@app.post("/generate")
def create_text(request: GenerationRequest):
    try:
        logger.info("Received text-generation request")
        return generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
    except Exception:
        logger.exception("Text generation failed")
        raise HTTPException(
            status_code=500,
            detail="Text generation failed. Please try again later.",
        ) from None

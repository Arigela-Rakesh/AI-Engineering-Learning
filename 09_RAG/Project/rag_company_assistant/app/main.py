
from contextlib import asynccontextmanager
from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag.pipeline import RAGPipeline
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCUMENT_PATH = Path("data/documents/company_policy.txt")

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG pipeline when the API starts."""
    global pipeline

    logger.info("Initializing RAG pipeline...")

    try:
        pipeline = RAGPipeline(
            document_path=str(DOCUMENT_PATH),
            chunk_size=25,
            overlap=5,
        )
        pipeline.ingest()

        logger.info(
            "RAG pipeline ready. Indexed %s chunks.",
            len(pipeline.chunks),
        )

        yield

    except Exception:
        logger.exception("Failed to initialize RAG pipeline.")
        raise

    finally:
        pipeline = None


app = FastAPI(
    title="Company Document Q&A API",
    description="Ask questions about company documents using RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="Question about the company documents",
    )
    top_k: int = Field(default=2, ge=1, le=5)
    min_score: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Optional minimum cosine similarity score",
    )


@app.get("/")
def root():
    return {
        "message": "Company Document Q&A API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "pipeline_ready": pipeline is not None,
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not ready.",
        )

    try:
        result = pipeline.answer(
            question=request.question,
            top_k=request.top_k,
            min_score=request.min_score,
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception("Failed to answer question.")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the question.",
        ) from exc

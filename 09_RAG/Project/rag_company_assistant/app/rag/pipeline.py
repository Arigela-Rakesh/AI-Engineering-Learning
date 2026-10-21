from pathlib import Path

from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_text
from app.retrieval.vector_store import VectorStore
from app.rag.generator import generate_answer


class RAGPipeline:
    def __init__(
        self,
        document_path: str,
        chunk_size: int = 60,
        overlap: int = 10,
    ):
        self.document_path = document_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.vector_store = VectorStore()
        self.chunks = []

    def ingest(self) -> None:
        """Load, chunk, embed, and index the document."""

        if not Path(self.document_path).is_file():
            raise FileNotFoundError(f"Document not found: {self.document_path}")

        records = load_document(self.document_path)
        documents = []

        for record in records:
            text_chunks = chunk_text(
                record["text"],
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )

            for chunk_id, chunk in enumerate(text_chunks):
                documents.append(
                    {
                        "text": chunk,
                        "source": record["source"],
                        "page": record["page"],
                        "chunk_id": chunk_id,
                    }
                )

        if not documents:
            raise ValueError("No chunks were created.")

        self.vector_store.add_documents(documents)
        self.chunks = documents

    def ask(
        self,
        question: str,
        top_k: int = 2,
    ) -> list[dict]:
        """Retrieve chunks relevant to a question."""

        if not self.chunks:
            raise RuntimeError("Pipeline is not initialized. Call ingest() first.")

        if not question or not question.strip():
            raise ValueError("Question must not be empty.")

        return self.vector_store.search(question, top_k=top_k)

    def answer(
        self,
        question: str,
        top_k: int = 2,
        min_score: float | None = None,
    ) -> dict:
        results = self.vector_store.search(
            question,
            top_k=top_k,
            min_score=min_score,
        )

        if not results:
            return {
                "question": question,
                "answer": (
                    "I couldn't find sufficiently relevant information "
                    "in the available documents to answer this question."
                ),
                "sources": [],
                "retrieved_chunks": [],
            }

        context_parts = []

        for result in results:
            source_label = result["source"]

            if result["page"] is not None:
                source_label += f", page {result['page']}"

            context_parts.append(f"Source: {source_label}\nContent: {result['text']}")

        context = "\n\n".join(context_parts)

        from app.rag.generator import generate_answer

        answer_text = generate_answer(
            question=question,
            context=context,
        )

        citations = []

        for result in results:
            citation = {
                "source": result["source"],
                "page": result["page"],
                "chunk_id": result["chunk_id"],
            }

            if citation not in citations:
                citations.append(citation)

        return {
            "question": question,
            "answer": answer_text,
            "sources": citations,
            "retrieved_chunks": results,
        }

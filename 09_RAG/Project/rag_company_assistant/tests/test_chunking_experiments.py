
from pathlib import Path

from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_text
from app.retrieval.vector_store import VectorStore
from app.retrieval.evaluation import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


DOCUMENT_PATH = Path("data/documents/company_policy.txt")
QUESTION = "What is the internet reimbursement limit?"
RELEVANT_IDS = {2}  # Correct for the existing 25-word configuration.


def evaluate_configuration(chunk_size, overlap, top_k=3):
    records = load_document(str(DOCUMENT_PATH))
    documents = []

    for record in records:
        chunks = chunk_text(
            record["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_id, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "source": record["source"],
                "page": record["page"],
                "chunk_id": chunk_id,
            })

    store = VectorStore()
    store.add_documents(documents)
    results = store.search(QUESTION, top_k=top_k)
    retrieved_ids = [result["chunk_id"] for result in results]

    # Show actual chunk text so relevance can be checked.
    print(f"\nChunk size={chunk_size}, overlap={overlap}, top_k={top_k}")
    for result in results:
        print(
            f"Chunk {result['chunk_id']} "
            f"(score={result['score']:.3f}): {result['text']}"
        )

    # These relevance labels are valid only for the original
    # chunking configuration. Don't score alternative chunkings
    # against old chunk IDs.
    if chunk_size == 25 and overlap == 5:
        print(
            "Precision@K:",
            round(precision_at_k(retrieved_ids, RELEVANT_IDS, top_k), 3),
        )
        print(
            "Recall@K:",
            round(recall_at_k(retrieved_ids, RELEVANT_IDS, top_k), 3),
        )
        print(
            "Reciprocal Rank:",
            round(reciprocal_rank(retrieved_ids, RELEVANT_IDS), 3),
        )


def main():
    for chunk_size, overlap in [(15, 3), (25, 5), (40, 8)]:
        evaluate_configuration(chunk_size, overlap)


if __name__ == "__main__":
    main()

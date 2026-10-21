
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
CHUNK_SIZE = 25
OVERLAP = 5
TOP_K = 3


def build_evaluation_store():
    records = load_document(str(DOCUMENT_PATH))
    documents = []

    for record in records:
        chunks = chunk_text(
            record["text"],
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP,
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

    return store, documents


def main():
    # Relevant chunk IDs are manually judged after inspecting
    # the document's chunks. Update these labels if needed.
    evaluation_cases = [
        {
            "question": "How many annual leave days do employees receive?",
            "relevant_ids": {0},
        },
        {
            "question": "What is the internet reimbursement limit?",
            "relevant_ids": {2},
        },
        {
            "question": "What is the company's maternity leave policy?",
            "relevant_ids": set(),
        },
    ]

    store, documents = build_evaluation_store()

    print(f"Total indexed chunks: {len(documents)}")
    print(f"Evaluating top K = {TOP_K}\n")

    precision_scores = []
    reciprocal_rank_scores = []

    for case in evaluation_cases:
        question = case["question"]
        relevant_ids = case["relevant_ids"]

        results = store.search(question, top_k=TOP_K)
        retrieved_ids = [result["chunk_id"] for result in results]

        print(f"Question: {question}")
        print(f"Retrieved chunk IDs: {retrieved_ids}")
        print(f"Expected relevant IDs: {sorted(relevant_ids)}")

        if relevant_ids:
            precision = precision_at_k(
                retrieved_ids, relevant_ids, TOP_K
            )
            recall = recall_at_k(
                retrieved_ids, relevant_ids, TOP_K
            )
            rr = reciprocal_rank(retrieved_ids, relevant_ids)

            precision_scores.append(precision)
            reciprocal_rank_scores.append(rr)

            print(f"Precision@{TOP_K}: {precision:.3f}")
            print(f"Recall@{TOP_K}: {recall:.3f}")
            print(f"Reciprocal Rank: {rr:.3f}")
        else:
            print(
                "Metrics skipped: no relevant chunk labels provided. "
                "Judge whether any retrieved chunks actually answer "
                "this question before scoring it."
            )

        for result in results:
            print(
                f"  Chunk {result['chunk_id']} "
                f"(similarity={result['score']:.3f}): "
                f"{result['text']}"
            )

        print("-" * 60)

    if precision_scores:
        mean_precision = sum(precision_scores) / len(precision_scores)
        mean_rr = sum(reciprocal_rank_scores) / len(reciprocal_rank_scores)

        print("\nSummary (labelled answerable questions only)")
        print(f"Mean Precision@{TOP_K}: {mean_precision:.3f}")
        print(f"Mean Reciprocal Rank: {mean_rr:.3f}")


if __name__ == "__main__":
    main()

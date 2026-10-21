
from pathlib import Path

from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_text
from app.retrieval.vector_store import VectorStore


document_path = Path("data/documents/company_policy.txt")

records = load_document(str(document_path))
documents = []

for record in records:
    chunks = chunk_text(
        record["text"],
        chunk_size=25,
        overlap=5,
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

question = "What is the internet reimbursement limit?"
results = store.search(question, top_k=3)

print("Retrieved context:\n")

for result in results:
    print(f"Chunk ID: {result['chunk_id']}")
    print(f"Similarity: {result['score']}")
    print(f"Text: {result['text']}")
    print("-" * 50)

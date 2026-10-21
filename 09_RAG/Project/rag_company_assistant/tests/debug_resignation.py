
from app.rag.pipeline import RAGPipeline

pipeline = RAGPipeline(
    document_path="data/documents/company_policy.txt",
    chunk_size=25,
    overlap=5,
)

pipeline.ingest()

question = "What is the resignation process for employees?"

results = pipeline.vector_store.search(
    question,
    top_k=2,
)

print("\n--- Retrieved context ---")

for result in results:
    print(f"\nChunk ID: {result['chunk_id']}")
    print(f"Score: {result['score']:.3f}")
    print(result["text"])

context_parts = []

for result in results:
    context_parts.append(
        f"Source: {result['source']}\n"
        f"Content: {result['text']}"
    )

context = "\n\n".join(context_parts)

print("\n--- Generator response ---")

from app.rag.generator import generate_answer

answer = generate_answer(
    question=question,
    context=context,
)

print("\nFinal answer:", answer)


from app.rag.pipeline import RAGPipeline


def main():
    pipeline = RAGPipeline(
        document_path="data/documents/company_policy.txt",
        chunk_size=25,
        overlap=5,
    )

    print("Preparing documents...")
    pipeline.ingest()

    test_cases = [
        {
            "question": "How many annual leave days do employees receive?",
            "expected_terms": ["20 days", "annual leave"],
        },
        {
            "question": "What is the internet reimbursement limit?",
            "expected_terms": ["1500 rupees", "per month"],
        },
        {
            "question": "How many days can employees work from home?",
            "expected_terms": ["two days per week", "work from home"],
        },
    ]

    k = 2
    hits = 0

    for test_case in test_cases:
        question = test_case["question"]
        expected_terms = test_case["expected_terms"]

        results = pipeline.ask(question, top_k=k)

        retrieved_text = " ".join(
            result["text"].lower()
            for result in results
        )

        hit = all(
            term.lower() in retrieved_text
            for term in expected_terms
        )

        if hit:
            hits += 1

        print(f"\nQuestion: {question}")
        print(f"Hit@{k}: {hit}")

        for result in results:
            print(f"Score: {result['score']}")
            print(f"Chunk: {result['text']}")

    hit_rate = hits / len(test_cases)

    print("\nEvaluation summary")
    print(f"Questions tested: {len(test_cases)}")
    print(f"Successful retrievals: {hits}")
    print(f"Hit@{k}: {hit_rate:.2f}")


if __name__ == "__main__":
    main()

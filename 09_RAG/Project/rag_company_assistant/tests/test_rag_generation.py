from app.rag.pipeline import RAGPipeline


def main():
    pipeline = RAGPipeline(
        document_path="data/documents/company_policy.txt",
        chunk_size=60,
        overlap=10,
    )

    print("Preparing company documents...")
    pipeline.ingest()

    print(f"Total chunks indexed: {len(pipeline.chunks)}")

    questions = [
        "How many annual leave days do employees receive?",
        "What is the internet reimbursement limit?",
        "What is the company's maternity leave policy?",
    ]

    for question in questions:
        result = pipeline.answer(question, top_k=2)

        print(f"\nQuestion: {question}")
        print(f"Answer: {result['answer']}")
        print("Citations:")

        for citation in result["sources"]:
            page_label = (
                f"page {citation['page']}"
                if citation["page"] is not None
                else "page not applicable"
            )

            print(
                f"- {citation['source']}, {page_label}, "
                f"chunk {citation['chunk_id']}"
            )

        print("-" * 60)


if __name__ == "__main__":
    main()

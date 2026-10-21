
from tests.test_top_k_experiment import build_store


def main():
    store = build_store()
    question = "What is the internet reimbursement limit?"

    for threshold in [None, 0.30, 0.50, 0.70]:
        results = store.search(
            question,
            top_k=5,
            min_score=threshold,
        )

        print(f"\nThreshold: {threshold}")
        print(f"Chunks retained: {len(results)}")

        for result in results:
            print(
                f"Chunk {result['chunk_id']} | "
                f"score={result['score']:.3f} | "
                f"{result['text']}"
            )


if __name__ == "__main__":
    main()

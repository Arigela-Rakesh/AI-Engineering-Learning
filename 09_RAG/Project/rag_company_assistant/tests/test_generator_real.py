
from app.rag.generator import generate_answer, FALLBACK_ANSWER


def main():
    test_cases = [
        {
            "question": "How many annual leave days do employees receive?",
            "context": (
                "Company Leave Policy: Employees receive 20 days "
                "of annual leave per calendar year."
            ),
        },
        {
            "question": "What is the internet reimbursement limit?",
            "context": (
                "The company reimburses eligible internet expenses "
                "up to 1500 rupees per month."
            ),
        },
        {
            "question": "What is the company's maternity leave policy?",
            "context": (
                "Employees receive 20 days of annual leave per "
                "calendar year. New employees must complete "
                "onboarding training during their first two weeks."
            ),
        },
    ]

    for index, case in enumerate(test_cases, start=1):
        print("=" * 60)
        print(f"Test {index}")
        print(f"Question: {case['question']}")

        answer = generate_answer(
            question=case["question"],
            context=case["context"],
        )

        print(f"Answer: {answer}")

        if index == 1:
            print(
                "Check: Does the answer state 20 annual leave days?"
            )
        elif index == 2:
            print(
                "Check: Does the answer state 1500 rupees per month?"
            )
        else:
            print(
                "Check: Does the answer avoid inventing a maternity "
                "leave policy?"
            )
            print(f"Expected fallback wording: {FALLBACK_ANSWER}")

    print("=" * 60)


if __name__ == "__main__":
    main()

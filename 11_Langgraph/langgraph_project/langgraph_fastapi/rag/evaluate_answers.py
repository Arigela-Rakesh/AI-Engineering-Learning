
import json
import re
from typing import Any

from langchain_ollama import ChatOllama

from rag.rag_pipeline import (
    initialize_vector_store,
    retrieve_context,
    generate_answer,
)


# ---------------------------------------------------------
# 1. Evaluation dataset
# ---------------------------------------------------------

TEST_CASES = [
    {
        "question": "How many annual leave days are provided?",
        "expected_keywords": ["20 days", "annual leave"],
        "should_have_context": True,
    },
    {
        "question": "What is the internet reimbursement amount?",
        "expected_keywords": ["1500", "per month"],
        "should_have_context": True,
    },
    {
        "question": "How many maternity leave days are provided?",
        "expected_keywords": [
            "does not specify",
            "consult HR",
        ],
        "should_have_context": True,
    },
    {
        "question": "What is the capital city of India?",
        "expected_keywords": [
            "does not contain relevant information",
        ],
        "should_have_context": False,
    },
]


# ---------------------------------------------------------
# 2. Deterministic evaluation
# ---------------------------------------------------------

def normalize_text(text: str) -> str:
    """
    Convert text to lowercase and remove punctuation.
    """
    text = text.lower()

    normalized = ""

    for character in text:
        if character.isalnum() or character.isspace():
            normalized += character
        else:
            normalized += " "

    return " ".join(normalized.split())


def check_expected_answer(
    answer: str,
    expected_keywords: list[str],
) -> bool:
    """
    Check whether all expected keywords are present.
    """

    normalized_answer = normalize_text(answer)

    for keyword in expected_keywords:
        normalized_keyword = normalize_text(keyword)

        if normalized_keyword not in normalized_answer:
            return False

    return True


# ---------------------------------------------------------
# 3. LLM-based evaluation
# ---------------------------------------------------------

def evaluate_answer_with_llm(
    question: str,
    context: str,
    answer: str,
) -> dict[str, Any]:
    """
    Evaluate correctness, groundedness, and hallucination.

    Scores:
        1.0 = Good
        0.0 = Poor

    Hallucination score:
        1.0 = No unsupported claims
        0.0 = Major unsupported claims
    """

    evaluator = ChatOllama(
        model="llama3.2:3b",
        temperature=0,
    )

    system_prompt = """
You are a strict evaluator for a Retrieval-Augmented Generation system.

Evaluate the generated answer using the question and retrieved context.

IMPORTANT RULES:

1. Correctness:
   - Score 1.0 if the answer correctly answers the question.
   - Score 1.0 if the answer appropriately says the context
     does not contain enough information.
   - Score 0.0 if the answer is incorrect or misleading.

2. Groundedness:
   - Score 1.0 if every factual claim is supported by the context.
   - Score 1.0 if the answer appropriately abstains because
     the context lacks the requested information.
   - Score 0.0 if the answer contains unsupported claims.

3. Hallucination:
   - Score 1.0 means no unsupported factual claims.
   - Score 0.0 means major unsupported factual claims.
   - A correct abstention receives 1.0.

Do not penalize an answer for being concise.

Return ONLY valid JSON in this exact format:

{
    "correctness": 0.0,
    "groundedness": 0.0,
    "hallucination": 0.0,
    "explanation": "Brief explanation"
}
"""

    user_prompt = f"""
Question:
{question}

Retrieved Context:
{context if context else "No relevant context retrieved."}

Generated Answer:
{answer}

Evaluate the answer using the rubric.
"""

    response = evaluator.invoke(
        [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    )

    response_text = response.content

    if not isinstance(response_text, str):
        response_text = str(response_text)

    # Extract JSON if the model adds markdown fences.
    match = re.search(
        r"\{.*\}",
        response_text,
        re.DOTALL,
    )

    if match is None:
        raise ValueError(
            f"Evaluator did not return valid JSON: {response_text}"
        )

    evaluation = json.loads(match.group())

    required_fields = [
        "correctness",
        "groundedness",
        "hallucination",
        "explanation",
    ]

    for field in required_fields:
        if field not in evaluation:
            raise ValueError(
                f"Missing evaluator field: {field}"
            )

    return evaluation


# ---------------------------------------------------------
# 4. Main evaluation process
# ---------------------------------------------------------

def main():

    print("Initializing vector store...")

    index, chunks = initialize_vector_store()

    print("Vector store initialized.")

    evaluator_results = []

    deterministic_passes = 0
    retrieval_passes = 0

    total_tests = len(TEST_CASES)

    for test_case in TEST_CASES:

        question = test_case["question"]

        expected_keywords = test_case["expected_keywords"]

        should_have_context = test_case["should_have_context"]

        print("\n" + "=" * 70)

        print(f"Question: {question}")

        # -------------------------------------------------
        # Retrieve relevant context
        # -------------------------------------------------

        context, retrieved_results = retrieve_context(
            question=question,
            index=index,
            chunks=chunks,
        )

        retrieved_count = len(retrieved_results)

        print(f"\nRetrieved Chunks: {retrieved_count}")

        # -------------------------------------------------
        # Retrieval evaluation
        # -------------------------------------------------

        if should_have_context:
            retrieval_pass = retrieved_count > 0
        else:
            retrieval_pass = retrieved_count == 0

        if retrieval_pass:
            retrieval_passes += 1

        print(
            f"Retrieval Check: "
            f"{'PASS' if retrieval_pass else 'FAIL'}"
        )

        # -------------------------------------------------
        # Generate answer
        # -------------------------------------------------

        if context:
            answer = generate_answer(
                question=question,
                context=context,
            )
        else:
            answer = (
                "The provided handbook does not contain "
                "relevant information to answer this question."
            )

        print("\nGenerated Answer:")
        print(answer)

        # -------------------------------------------------
        # Deterministic evaluation
        # -------------------------------------------------

        deterministic_pass = check_expected_answer(
            answer=answer,
            expected_keywords=expected_keywords,
        )

        if deterministic_pass:
            deterministic_passes += 1

        print(
            "\nDeterministic Check: "
            f"{'PASS' if deterministic_pass else 'FAIL'}"
        )

        # -------------------------------------------------
        # LLM evaluation
        # -------------------------------------------------

        evaluation = evaluate_answer_with_llm(
            question=question,
            context=context,
            answer=answer,
        )

        print("\nEvaluation Scores:")

        print(
            f"Correctness: "
            f"{evaluation['correctness']}"
        )

        print(
            f"Groundedness: "
            f"{evaluation['groundedness']}"
        )

        print(
            f"Hallucination: "
            f"{evaluation['hallucination']}"
        )

        print(
            f"Explanation: "
            f"{evaluation['explanation']}"
        )

        evaluator_results.append(evaluation)

    # -----------------------------------------------------
    # 5. Final summary
    # -----------------------------------------------------

    average_correctness = (
        sum(
            result["correctness"]
            for result in evaluator_results
        )
        / total_tests
    )

    average_groundedness = (
        sum(
            result["groundedness"]
            for result in evaluator_results
        )
        / total_tests
    )

    average_hallucination = (
        sum(
            result["hallucination"]
            for result in evaluator_results
        )
        / total_tests
    )

    deterministic_pass_rate = (
        deterministic_passes / total_tests
    ) * 100

    retrieval_pass_rate = (
        retrieval_passes / total_tests
    ) * 100

    print("\n" + "=" * 70)

    print("RAG EVALUATION SUMMARY")

    print("=" * 70)

    print(
        f"Total Test Cases: {total_tests}"
    )

    print(
        f"Retrieval Pass Rate: "
        f"{retrieval_pass_rate:.2f}%"
    )

    print(
        f"Deterministic Answer Pass Rate: "
        f"{deterministic_pass_rate:.2f}%"
    )

    print(
        f"Average Correctness: "
        f"{average_correctness:.2f}"
    )

    print(
        f"Average Groundedness: "
        f"{average_groundedness:.2f}"
    )

    print(
        f"Average Hallucination Score: "
        f"{average_hallucination:.2f}"
    )

    print("=" * 70)


# ---------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()

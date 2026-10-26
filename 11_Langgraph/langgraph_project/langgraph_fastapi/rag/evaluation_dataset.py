
"""
Lesson 132: RAG Evaluation Dataset

This dataset tests:
1. Direct retrieval
2. Paraphrased questions
3. Missing information
4. Out-of-domain questions
5. Similar but irrelevant questions

Update expected_keywords only when the handbook supports
the expected answer.
"""

EVALUATION_DATASET = [
    # -----------------------------------------------------
    # Category 1: Direct retrieval
    # -----------------------------------------------------
    {
        "id": "direct_001",
        "category": "direct",
        "question": "How many annual leave days are provided?",
        "expected_keywords": ["20 days"],
        "should_retrieve": True,
    },
    {
        "id": "direct_002",
        "category": "direct",
        "question": "What is the internet reimbursement amount?",
        "expected_keywords": ["1500", "per month"],
        "should_retrieve": True,
    },

    # -----------------------------------------------------
    # Category 2: Paraphrased questions
    # -----------------------------------------------------
    {
        "id": "paraphrase_001",
        "category": "paraphrase",
        "question": "How many days of annual vacation can employees take?",
        "expected_keywords": ["20 days"],
        "should_retrieve": True,
    },
    {
        "id": "paraphrase_002",
        "category": "paraphrase",
        "question": "How much money is provided monthly for internet expenses?",
        "expected_keywords": ["1500", "per month"],
        "should_retrieve": True,
    },

    # -----------------------------------------------------
    # Category 3: Missing information
    # -----------------------------------------------------
    {
        "id": "missing_001",
        "category": "missing_information",
        "question": "How many maternity leave days are provided?",
        "expected_keywords": [
            "does not specify",
            "consult HR",
        ],
        "should_retrieve": True,
    },

    # -----------------------------------------------------
    # Category 4: Out-of-domain questions
    # -----------------------------------------------------
    {
        "id": "ood_001",
        "category": "out_of_domain",
        "question": "What is the capital city of India?",
        "expected_keywords": [
            "does not contain relevant information",
        ],
        "should_retrieve": False,
    },

    # -----------------------------------------------------
    # Category 5: Similar but irrelevant questions
    # -----------------------------------------------------
    {
        "id": "irrelevant_001",
        "category": "irrelevant",
        "question": "Does the handbook provide a home loan?",
        "expected_keywords": [
            "does not contain relevant information",
        ],
        "should_retrieve": False,
    },
    {
        "id": "irrelevant_002",
        "category": "irrelevant",
        "question": "Does the company reimburse employees for car purchases?",
        "expected_keywords": [
            "does not contain relevant information",
        ],
        "should_retrieve": True,
    },
]

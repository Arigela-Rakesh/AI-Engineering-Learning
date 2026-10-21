
from unittest.mock import Mock, patch

import pytest
import torch

from app.rag.generator import (
    FALLBACK_ANSWER,
    generate_answer,
)


def test_rejects_empty_question():
    with pytest.raises(ValueError, match="Question must not be empty"):
        generate_answer("", "Some document context")


def test_rejects_empty_context():
    with pytest.raises(ValueError, match="Context must not be empty"):
        generate_answer("What is the leave policy?", "  ")


def test_rejects_invalid_max_new_tokens():
    with pytest.raises(
        ValueError,
        match="max_new_tokens must be greater than zero",
    ):
        generate_answer(
            "What is the leave policy?",
            "Employees receive 20 days of annual leave.",
            max_new_tokens=0,
        )


def test_generates_answer_from_context():
    fake_tokenizer = Mock()
    fake_tokenizer.apply_chat_template.return_value = "formatted prompt"
    fake_tokenizer.return_value = {
        "input_ids": torch.tensor([[10, 20]])
    }
    fake_tokenizer.eos_token_id = 0
    fake_tokenizer.decode.return_value = (
        "Employees receive 20 days of annual leave."
    )

    fake_model = Mock()
    fake_model.generate.return_value = torch.tensor([[10, 20, 30, 40]])

    with (
        patch("app.rag.generator.load_generator"),
        patch("app.rag.generator._tokenizer", fake_tokenizer),
        patch("app.rag.generator._model", fake_model),
    ):
        answer = generate_answer(
            "How many annual leave days do employees receive?",
            "Employees receive 20 days of annual leave.",
        )

    assert answer == "Employees receive 20 days of annual leave."
    fake_model.generate.assert_called_once()


def test_returns_fallback_for_empty_model_output():
    fake_tokenizer = Mock()
    fake_tokenizer.apply_chat_template.return_value = "formatted prompt"
    fake_tokenizer.return_value = {
        "input_ids": torch.tensor([[10, 20]])
    }
    fake_tokenizer.eos_token_id = 0
    fake_tokenizer.decode.return_value = "   "

    fake_model = Mock()
    fake_model.generate.return_value = torch.tensor([[10, 20, 30]])

    with (
        patch("app.rag.generator.load_generator"),
        patch("app.rag.generator._tokenizer", fake_tokenizer),
        patch("app.rag.generator._model", fake_model),
    ):
        answer = generate_answer(
            "What is the leave policy?",
            "The document mentions annual leave.",
        )

    assert answer == FALLBACK_ANSWER

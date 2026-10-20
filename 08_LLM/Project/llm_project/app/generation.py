
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from app.config import (
    GENERATION_MODEL_NAME,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
)

tokenizer = None
model = None


def load_generation_model() -> None:
    global tokenizer, model

    if model is not None and tokenizer is not None:
        return

    loaded_tokenizer = AutoTokenizer.from_pretrained(
        GENERATION_MODEL_NAME
    )

    loaded_model = AutoModelForCausalLM.from_pretrained(
        GENERATION_MODEL_NAME
    )

    if loaded_tokenizer.pad_token is None:
        loaded_tokenizer.pad_token = loaded_tokenizer.eos_token

    loaded_model.config.pad_token_id = (
        loaded_tokenizer.pad_token_id
    )
    loaded_model.eval()

    tokenizer = loaded_tokenizer
    model = loaded_model


def generate_text(
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    if model is None or tokenizer is None:
        raise RuntimeError("Generation model is not loaded")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. "
                "Answer questions directly and concisely. "
                "If you are unsure, say so instead of guessing."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][input_length:]

    generated_text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    return {
        "prompt": prompt,
        "generated_text": generated_text,
        "settings": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        },
    }

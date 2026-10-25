
import json
from typing import TypeVar

import ollama
from pydantic import BaseModel, ValidationError

from src.config import EMBED_MODEL, GEN_MODEL

T = TypeVar("T", bound=BaseModel)


def chat(
    prompt: str,
    system: str = "You are a helpful assistant. Be accurate and concise.",
    model: str = GEN_MODEL,
    temperature: float = 0.0,
) -> str:
    """Generate a text response using a local Ollama model."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": temperature},
    )

    return response.message.content or ""



def chat_json(
    prompt: str,
    schema: type[T],
    system: str = "Return a response that matches the required JSON schema.",
    model: str = GEN_MODEL,
) -> T:
    """Generate and validate structured output, retrying once if invalid."""
    repair_prompt = prompt
    last_error = None

    for attempt in range(2):
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": repair_prompt},
            ],
            format=schema.model_json_schema(),
            options={"temperature": 0.0},
        )

        raw = response.message.content or "{}"

        try:
            return schema.model_validate_json(raw)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == 0:
                repair_prompt = (
                    f"{prompt}\n\n"
                    "Your previous response was invalid and could not be "
                    "validated against the required JSON schema. "
                    "Return a complete, concise, valid JSON object only. "
                    "Do not truncate the response or add commentary."
                )

    raise ValueError(
        f"Model returned invalid structured output after 2 attempts: {raw}"
    ) from last_error


def embed(texts: list[str]) -> list[list[float]]:
    """Generate local embedding vectors for one or more texts."""
    if not texts:
        return []
    response = ollama.embed(
        model=EMBED_MODEL,
        input=texts,
    )

    return response.embeddings

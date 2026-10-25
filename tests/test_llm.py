from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from src import llm


class ExampleSchema(BaseModel):
    answer: str
    score: int


def mock_chat_response(content):
    return SimpleNamespace(
        message=SimpleNamespace(content=content)
    )


def test_chat_returns_model_response(monkeypatch):
    ollama_chat = Mock(
        return_value=mock_chat_response("Hello from the model")
    )
    monkeypatch.setattr(llm.ollama, "chat", ollama_chat)

    result = llm.chat(
        prompt="Say hello",
        system="Be concise",
        model="test-model",
        temperature=0.4,
    )

    assert result == "Hello from the model"
    ollama_chat.assert_called_once_with(
        model="test-model",
        messages=[
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Say hello"},
        ],
        options={"temperature": 0.4},
    )


def test_chat_returns_empty_string_for_empty_content(monkeypatch):
    monkeypatch.setattr(
        llm.ollama,
        "chat",
        Mock(return_value=mock_chat_response(None)),
    )

    assert llm.chat("Say hello") == ""


def test_chat_uses_default_arguments(monkeypatch):
    ollama_chat = Mock(
        return_value=mock_chat_response("Default response")
    )
    monkeypatch.setattr(llm.ollama, "chat", ollama_chat)

    result = llm.chat("Test prompt")

    assert result == "Default response"
    kwargs = ollama_chat.call_args.kwargs
    assert kwargs["model"] == llm.GEN_MODEL
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][1]["content"] == "Test prompt"
    assert kwargs["options"] == {"temperature": 0.0}


def test_chat_json_returns_validated_schema(monkeypatch):
    ollama_chat = Mock(
        return_value=mock_chat_response(
            '{"answer": "42", "score": 5}'
        )
    )
    monkeypatch.setattr(llm.ollama, "chat", ollama_chat)

    result = llm.chat_json(
        prompt="Answer the question",
        schema=ExampleSchema,
        system="Return JSON",
        model="test-model",
    )

    assert isinstance(result, ExampleSchema)
    assert result.answer == "42"
    assert result.score == 5

    ollama_chat.assert_called_once_with(
        model="test-model",
        messages=[
            {"role": "system", "content": "Return JSON"},
            {"role": "user", "content": "Answer the question"},
        ],
        format=ExampleSchema.model_json_schema(),
        options={"temperature": 0.0},
    )


def test_chat_json_retries_invalid_output_then_succeeds(monkeypatch):
    ollama_chat = Mock(
        side_effect=[
            mock_chat_response('{"answer": "42"'),
            mock_chat_response('{"answer": "42", "score": 5}'),
        ]
    )
    monkeypatch.setattr(llm.ollama, "chat", ollama_chat)

    result = llm.chat_json("Question", ExampleSchema)

    assert result.answer == "42"
    assert result.score == 5
    assert ollama_chat.call_count == 2
    second_prompt = ollama_chat.call_args_list[1].kwargs["messages"][1]["content"]
    assert "previous response was invalid" in second_prompt


def test_chat_json_raises_after_two_invalid_responses(monkeypatch):
    ollama_chat = Mock(
        side_effect=[
            mock_chat_response('{"answer": "42"'),
            mock_chat_response('{"answer": "42"'),
        ]
    )
    monkeypatch.setattr(llm.ollama, "chat", ollama_chat)

    with pytest.raises(
        ValueError,
        match="invalid structured output after 2 attempts",
    ):
        llm.chat_json("Question", ExampleSchema)

    assert ollama_chat.call_count == 2



def test_chat_json_uses_empty_object_when_content_is_none(monkeypatch):
    monkeypatch.setattr(
        llm.ollama,
        "chat",
        Mock(return_value=mock_chat_response(None)),
    )

    with pytest.raises(
        ValueError,
        match="Model returned invalid structured output",
    ):
        llm.chat_json("Question", ExampleSchema)


@pytest.mark.parametrize(
    "invalid_json",
    [
        '{"answer": "42", "score": "not-an-integer"}',
        '{"answer": "42"',
        '{"unexpected": "field"}',
    ],
)
def test_chat_json_rejects_invalid_output(monkeypatch, invalid_json):
    monkeypatch.setattr(
        llm.ollama,
        "chat",
        Mock(return_value=mock_chat_response(invalid_json)),
    )

    with pytest.raises(
        ValueError,
        match="Model returned invalid structured output",
    ):
        llm.chat_json("Question", ExampleSchema)


def test_embed_returns_empty_for_empty_input(monkeypatch):
    ollama_embed = Mock()
    monkeypatch.setattr(llm.ollama, "embed", ollama_embed)

    assert llm.embed([]) == []
    ollama_embed.assert_not_called()


def test_embed_returns_vectors_from_ollama(monkeypatch):
    vectors = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    ollama_embed = Mock(
        return_value=SimpleNamespace(embeddings=vectors)
    )
    monkeypatch.setattr(llm.ollama, "embed", ollama_embed)

    result = llm.embed(["first text", "second text"])

    assert result == vectors
    ollama_embed.assert_called_once_with(
        model=llm.EMBED_MODEL,
        input=["first text", "second text"],
    )

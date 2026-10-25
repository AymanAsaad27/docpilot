import json
from pathlib import Path
from unittest.mock import Mock

from src import agent
from src.judge import ClaimCheck, Verification


def test_generate_answer_returns_stripped_response(monkeypatch):
    chat = Mock(return_value="  Answer with citation [1].\n")
    monkeypatch.setattr(agent, "chat", chat)

    result = agent.generate_answer(
        "How does it work?",
        "[1] Source: guide.md\nSection: Usage\nExample documentation.",
    )

    assert result == "Answer with citation [1]."

    kwargs = chat.call_args.kwargs
    assert "How does it work?" in kwargs["prompt"]
    assert "Example documentation." in kwargs["prompt"]
    assert kwargs["temperature"] == 0.0


def test_generate_answer_includes_verification_feedback(monkeypatch):
    chat = Mock(return_value="Corrected answer [1].")
    monkeypatch.setattr(agent, "chat", chat)

    result = agent.generate_answer(
        "What is supported?",
        "[1] Source: guide.md\nSection: Usage\nVerified fact.",
        previous_answer="An unsupported answer.",
        verification_feedback="- Unsupported claim: extra feature.",
    )

    assert result == "Corrected answer [1]."
    prompt = chat.call_args.kwargs["prompt"]
    assert "An unsupported answer." in prompt
    assert "extra feature" in prompt
    assert "did not pass verification" in prompt


def test_build_verification_feedback_with_unsupported_claims():
    verification = Verification(
        claims=[
            ClaimCheck(
                claim="Supported claim",
                relevant=True,
                supported=True,
                evidence="Evidence found",
                evidence_quote="",
            ),
            ClaimCheck(
                claim="Unsupported claim",
                relevant=True,
                supported=False,
                evidence="No matching passage",
                evidence_quote="",
            ),
        ]
    )

    result = agent.build_verification_feedback(verification)

    assert result == (
        "- Unsupported claim: Unsupported claim. "
        "Evidence/reason: No matching passage"
    )


def test_build_verification_feedback_when_all_claims_supported():
    verification = Verification(
        claims=[
            ClaimCheck(
                claim="A fact",
                relevant=True,
                supported=True,
                evidence="The context supports it.",
                evidence_quote="",
            )
        ]
    )

    assert (
        agent.build_verification_feedback(verification)
        == "No unsupported claims were identified."
    )


def test_log_trace_appends_jsonl_record(monkeypatch, tmp_path):
    log_file = tmp_path / "logs" / "traces.jsonl"
    monkeypatch.setattr(agent, "LOG_FILE", log_file)

    agent.log_trace({"question": "First", "answer": "Café"})
    agent.log_trace({"question": "Second", "answer": "Done"})

    lines = log_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0]) == {
        "question": "First",
        "answer": "Café",
    }
    assert json.loads(lines[1]) == {
        "question": "Second",
        "answer": "Done",
    }

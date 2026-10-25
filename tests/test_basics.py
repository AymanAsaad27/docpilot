import pytest
from pydantic import BaseModel

from src.agent import format_context, run_agent
from src.judge import ClaimCheck, Verification, faithfulness_score


def test_empty_question_is_rejected():
    with pytest.raises(ValueError, match="Question must not be empty"):
        run_agent("   ")


def test_format_context_includes_source_and_heading():
    passages = [
        {
            "metadata": {
                "source": "tutorial/query-params.md",
                "heading": "Optional parameters",
            },
            "text": "Set the default value to None.",
        }
    ]

    context = format_context(passages)

    assert "[1] Source: tutorial/query-params.md" in context
    assert "Section: Optional parameters" in context
    assert "Set the default value to None." in context


def test_faithfulness_score_with_supported_and_unsupported_claims():
    verification = Verification(
        claims=[
            ClaimCheck(
                claim="Claim A",
                relevant=True,
                supported=True,
                evidence="The context supports Claim A.",
                evidence_quote="",
            ),
            ClaimCheck(
                claim="Claim B",
                relevant=True,
                supported=False,
                evidence="The context does not support Claim B.",
                evidence_quote="",
            ),
        ]
    )

    assert faithfulness_score(verification) == 0.5


def test_faithfulness_score_when_all_claims_are_supported():
    verification = Verification(
        claims=[
            ClaimCheck(
                claim="Claim A",
                relevant=True,
                supported=True,
                evidence="The context supports Claim A.",
                evidence_quote="",
            )
        ]
    )

    assert faithfulness_score(verification) == 1.0

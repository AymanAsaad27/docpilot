from unittest.mock import patch

import pytest

from src.agent import run_agent
from src.judge import ClaimCheck, Grade, Score, Verification


def make_passage(chunk_id="chunk-1", score=0.9):
    return {
        "id": chunk_id,
        "text": "FastAPI supports query parameters with default values.",
        "metadata": {
            "source": "tutorial/query-params.md",
            "heading": "Default values",
        },
        "rerank_score": score,
    }


def supported_verification():
    return Verification(
        claims=[
            ClaimCheck(
                claim="FastAPI supports default values.",
                relevant=True,
                supported=True,
                evidence="The context describes default values.",
                evidence_quote="",
            )
        ]
    )


def make_score(correctness=5, relevance=5):
    return Score(
        relevance=relevance,
        correctness=correctness,
        explanation="The answer is supported by the supplied context.",
    )


@patch("src.agent.log_trace")
@patch("src.agent.score_answer")
@patch("src.agent.verify_claims")
@patch("src.agent.generate_answer")
@patch("src.agent.grade_context")
@patch("src.agent.retrieve_for_queries")
@patch("src.agent.rewrite_queries")
def test_run_agent_completes_without_retry(
    mock_rewrite,
    mock_retrieve,
    mock_grade,
    mock_generate,
    mock_verify,
    mock_score,
    mock_log,
):
    mock_rewrite.return_value = ["FastAPI default query parameters"]
    mock_retrieve.return_value = [make_passage()]
    mock_grade.return_value = Grade(
        sufficient=True,
        reason="The context explains default values.",
    )
    mock_generate.return_value = "Use a default value. [1]"
    mock_verify.return_value = supported_verification()
    mock_score.return_value = make_score()

    result = run_agent("How do query parameter defaults work?")

    assert result["answer"] == "Use a default value. [1]"
    assert result["context_sufficient"] is True
    assert result["retries"] == 0
    assert result["regenerations"] == 0
    assert result["low_confidence"] is False
    assert result["citations"][0]["source"] == "tutorial/query-params.md"
    mock_log.assert_called_once()


@patch("src.agent.log_trace")
@patch("src.agent.search")
@patch("src.agent.score_answer")
@patch("src.agent.verify_claims")
@patch("src.agent.generate_answer")
@patch("src.agent.grade_context")
@patch("src.agent.retrieve_for_queries")
@patch("src.agent.rewrite_queries")
def test_run_agent_retries_when_context_is_initially_insufficient(
    mock_rewrite,
    mock_retrieve,
    mock_grade,
    mock_generate,
    mock_verify,
    mock_score,
    mock_search,
    mock_log,
):
    mock_rewrite.return_value = []
    mock_retrieve.return_value = [make_passage("initial", 0.5)]
    mock_grade.side_effect = [
        Grade(sufficient=False, reason="Need a more specific example."),
        Grade(sufficient=True, reason="The additional context is sufficient."),
    ]
    mock_search.return_value = [make_passage("additional", 0.9)]
    mock_generate.return_value = "Use a default value. [1]"
    mock_verify.return_value = supported_verification()
    mock_score.return_value = make_score()

    result = run_agent("How do query parameter defaults work?")

    assert result["retries"] == 1
    assert result["context_sufficient"] is True
    assert mock_search.call_count == 1
    assert mock_grade.call_count == 2
    assert len(result["citations"]) == 2
    mock_log.assert_called_once()


@patch("src.agent.log_trace")
@patch("src.agent.score_answer")
@patch("src.agent.verify_claims")
@patch("src.agent.generate_answer")
@patch("src.agent.grade_context")
@patch("src.agent.retrieve_for_queries")
@patch("src.agent.rewrite_queries")
def test_run_agent_regenerates_answer_when_faithfulness_is_low(
    mock_rewrite,
    mock_retrieve,
    mock_grade,
    mock_generate,
    mock_verify,
    mock_score,
    mock_log,
):
    mock_rewrite.return_value = []
    mock_retrieve.return_value = [make_passage()]
    mock_grade.return_value = Grade(
        sufficient=True,
        reason="The context is sufficient.",
    )

    mock_generate.side_effect = [
        "The endpoint supports feature X.",
        "The endpoint supports default values. [1]",
    ]

    mock_verify.side_effect = [
        Verification(
            claims=[
                ClaimCheck(
                    claim="The endpoint supports feature X.",
                    relevant=True,
                supported=False,
                    evidence="The context does not mention feature X.",
                    evidence_quote="",
                )
            ]
        ),
        supported_verification(),
    ]

    mock_score.return_value = make_score()

    result = run_agent("How do query parameter defaults work?")

    assert result["regenerations"] == 1
    assert mock_generate.call_count == 2
    assert result["faithfulness"] == 1.0
    assert result["low_confidence"] is False
    assert mock_generate.call_args.kwargs["previous_answer"] == (
        "The endpoint supports feature X."
    )
    mock_log.assert_called_once()


@patch("src.agent.log_trace")
@patch("src.agent.rewrite_queries", side_effect=RuntimeError("mock failure"))
def test_run_agent_logs_and_reraises_unexpected_errors(
    mock_rewrite,
    mock_log,
):
    with pytest.raises(RuntimeError, match="mock failure"):
        run_agent("A valid test question")

    mock_log.assert_called_once()
    trace = mock_log.call_args.args[0]

    assert trace["events"][-1]["stage"] == "error"
    assert trace["events"][-1]["error_type"] == "RuntimeError"
    assert trace["events"][-1]["error"] == "mock failure"

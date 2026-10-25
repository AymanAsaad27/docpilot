
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from src import judge


def test_grade_context_returns_grade_and_uses_judge_model(monkeypatch):
    expected = judge.Grade(
        sufficient=True,
        reason="The context explains the required configuration.",
    )
    chat_json = Mock(return_value=expected)
    monkeypatch.setattr(judge, "chat_json", chat_json)

    result = judge.grade_context(
        "How do I configure it?",
        "Set the value in the configuration file.",
    )

    assert result is expected
    kwargs = chat_json.call_args.kwargs
    assert kwargs["schema"] is judge.Grade
    assert kwargs["model"] == judge.JUDGE_MODEL
    assert "How do I configure it?" in kwargs["prompt"]
    assert "Set the value in the configuration file." in kwargs["prompt"]
    assert "outside knowledge" in kwargs["prompt"].lower()


def test_verify_claims_returns_verification(monkeypatch):
    expected = judge.Verification(
        claims=[
            judge.ClaimCheck(
                claim="FastAPI supports dependency injection.",
                relevant=True,
                supported=True,
                evidence="The context describes dependency injection.",
                evidence_quote="",
            ),
            judge.ClaimCheck(
                claim="FastAPI supports an unrelated feature.",
                relevant=False,
                supported=False,
                evidence="The context does not mention this feature.",
                evidence_quote="",
            ),
        ]
    )
    chat_json = Mock(return_value=expected)
    monkeypatch.setattr(judge, "chat_json", chat_json)

    result = judge.verify_claims(
        "What does FastAPI support?",
        "It supports dependency injection and another feature.",
        "The context describes dependency injection.",
    )

    assert result is expected
    assert len(result.claims) == 2
    assert result.claims[0].relevant is True
    assert result.claims[1].relevant is False

    kwargs = chat_json.call_args.kwargs
    assert kwargs["schema"] is judge.Verification
    assert kwargs["model"] == judge.JUDGE_MODEL
    assert "What does FastAPI support?" in kwargs["prompt"]
    assert "dependency injection and another feature" in kwargs["prompt"]
    assert "empty claims list" in kwargs["prompt"]
    assert "relevant=true" in kwargs["prompt"]
    assert "assess support independently of relevance" in kwargs["prompt"].lower()


def test_score_answer_returns_score_and_uses_judge_model(monkeypatch):
    expected = judge.Score(
        relevance=5,
        correctness=4,
        explanation="The answer is relevant and mostly supported.",
    )
    chat_json = Mock(return_value=expected)
    monkeypatch.setattr(judge, "chat_json", chat_json)

    result = judge.score_answer(
        "What is dependency injection?",
        "It supplies dependencies to a function.",
        "FastAPI supports dependency injection.",
    )

    assert result is expected
    kwargs = chat_json.call_args.kwargs
    assert kwargs["schema"] is judge.Score
    assert kwargs["model"] == judge.JUDGE_MODEL
    assert "What is dependency injection?" in kwargs["prompt"]
    assert "It supplies dependencies to a function." in kwargs["prompt"]
    assert "Do not reward plausible statements" in kwargs["prompt"]


@pytest.mark.parametrize(
    "field_values",
    [
        {"relevance": 0, "correctness": 3, "explanation": "Invalid"},
        {"relevance": 6, "correctness": 3, "explanation": "Invalid"},
        {"relevance": 3, "correctness": 0, "explanation": "Invalid"},
        {"relevance": 3, "correctness": 6, "explanation": "Invalid"},
    ],
)
def test_score_rejects_ratings_outside_one_to_five(field_values):
    with pytest.raises(ValidationError):
        judge.Score(**field_values)


def test_faithfulness_score_is_one_when_there_are_no_claims():
    verification = judge.Verification(claims=[])

    assert judge.faithfulness_score(verification) == 1.0


def test_faithfulness_score_is_one_when_all_claims_are_supported():
    verification = judge.Verification(
        claims=[
            judge.ClaimCheck(
                claim="Claim one",
                relevant=True,
                supported=True,
                evidence="Evidence one",
                evidence_quote="",
            ),
            judge.ClaimCheck(
                claim="Claim two",
                relevant=True,
                supported=True,
                evidence="Evidence two",
                evidence_quote="",
            ),
        ]
    )

    assert judge.faithfulness_score(verification) == 1.0


def test_faithfulness_score_is_zero_when_no_claims_are_supported():
    verification = judge.Verification(
        claims=[
            judge.ClaimCheck(
                claim="Claim one",
                relevant=True,
                supported=False,
                evidence="No supporting evidence",
                evidence_quote="",
            ),
            judge.ClaimCheck(
                claim="Claim two",
                relevant=True,
                supported=False,
                evidence="No supporting evidence",
                evidence_quote="",
            ),
        ]
    )

    assert judge.faithfulness_score(verification) == 0.0


def test_faithfulness_score_calculates_supported_fraction():
    verification = judge.Verification(
        claims=[
            judge.ClaimCheck(
                claim="Supported claim",
                relevant=True,
                supported=True,
                evidence="Matching context",
                evidence_quote="",
            ),
            judge.ClaimCheck(
                claim="Unsupported claim",
                relevant=True,
                supported=False,
                evidence="No evidence",
                evidence_quote="",
            ),
            judge.ClaimCheck(
                claim="Another supported claim",
                relevant=True,
                supported=True,
                evidence="Matching context",
                evidence_quote="",
            ),
            judge.ClaimCheck(
                claim="Another unsupported claim",
                relevant=True,
                supported=False,
                evidence="No evidence",
                evidence_quote="",
            ),
        ]
    )

    assert judge.faithfulness_score(verification) == 0.5


def test_faithfulness_score_ignores_irrelevant_claims():
    verification = judge.Verification(
        claims=[
            judge.ClaimCheck(
                claim="The answer addresses the question.",
                relevant=True,
                supported=True,
                evidence="The context supports this claim.",
                evidence_quote="",
            ),
            judge.ClaimCheck(
                claim="An unrelated fact from the documents.",
                relevant=False,
                supported=False,
                evidence="This claim is unrelated to the question.",
                evidence_quote="",
            ),
            judge.ClaimCheck(
                claim="Another relevant claim.",
                relevant=True,
                supported=False,
                evidence="The context does not support this claim.",
                evidence_quote="",
            ),
        ]
    )

    assert judge.faithfulness_score(verification) == 0.5


def test_faithfulness_score_is_one_when_all_claims_are_irrelevant():
    verification = judge.Verification(
        claims=[
            judge.ClaimCheck(
                claim="An unrelated claim.",
                relevant=False,
                supported=False,
                evidence="This claim is irrelevant.",
                evidence_quote="",
            )
        ]
    )

    assert judge.faithfulness_score(verification) == 1.0


def test_verify_cited_claims_uses_citation_specific_evidence(monkeypatch):
    expected = judge.Verification(
        claims=[
            judge.ClaimCheck(
                claim="FastAPI uses Pydantic for request validation.",
                relevant=True,
                supported=True,
                evidence="Passage 1 describes Pydantic-based request validation.",
                evidence_quote="",
            )
        ]
    )
    chat_json = Mock(return_value=expected)
    monkeypatch.setattr(judge, "chat_json", chat_json)

    result = judge.verify_cited_claims(
        question="How does FastAPI validate request data?",
        answer="FastAPI uses Pydantic for request validation [1].",
        citations=[
            {
                "citation": 1,
                "source": "tutorial/body.md",
                "heading": "Request Body",
                "text": "FastAPI uses Pydantic models to validate request bodies.",
            }
        ],
    )

    assert result is expected
    kwargs = chat_json.call_args.kwargs
    assert kwargs["schema"] is judge.Verification
    assert kwargs["model"] == judge.JUDGE_MODEL
    assert "request validation" in kwargs["prompt"].lower()
    assert "Pydantic models" in kwargs["prompt"]
    assert "passage or passages" in kwargs["prompt"].lower()
    assert "relevant=true" in kwargs["prompt"]


def test_verify_cited_claims_can_flag_unsupported_citation(monkeypatch):
    expected = judge.Verification(
        claims=[
            judge.ClaimCheck(
                claim="FastAPI automatically encrypts all request bodies.",
                relevant=True,
                supported=False,
                evidence="The cited passage discusses validation, not encryption.",
                evidence_quote="",
            )
        ]
    )
    chat_json = Mock(return_value=expected)
    monkeypatch.setattr(judge, "chat_json", chat_json)

    result = judge.verify_cited_claims(
        question="How does FastAPI validate request data?",
        answer="FastAPI automatically encrypts all request bodies [1].",
        citations=[
            {
                "citation": 1,
                "source": "tutorial/body.md",
                "heading": "Request Body",
                "text": "FastAPI uses Pydantic models to validate request bodies.",
            }
        ],
    )

    assert len(result.claims) == 1
    assert result.claims[0].relevant is True
    assert result.claims[0].supported is False
    assert "not encryption" in result.claims[0].evidence


def test_validate_evidence_quotes_keeps_a_verifiable_quote():
    verification = judge.Verification(
        claims=[
            judge.ClaimCheck(
                claim="FastAPI supports dependency injection.",
                relevant=True,
                supported=True,
                evidence="The context describes dependency injection.",
                evidence_quote="FastAPI supports dependency injection.",
            )
        ]
    )
    result = judge.validate_evidence_quotes(
        verification, "FastAPI supports dependency injection."
    )
    assert result.claims[0].supported is True
    assert result.claims[0].evidence_quote == (
        "FastAPI supports dependency injection."
    )


def test_validate_evidence_quotes_rejects_an_invented_quote():
    verification = judge.Verification(
        claims=[
            judge.ClaimCheck(
                claim="FastAPI supports feature X.",
                relevant=True,
                supported=True,
                evidence="The documentation allegedly supports this.",
                evidence_quote="FastAPI supports feature X.",
            )
        ]
    )
    result = judge.validate_evidence_quotes(
        verification, "FastAPI supports dependency injection."
    )
    assert result.claims[0].supported is False
    assert result.claims[0].evidence_quote == ""
    assert "deterministic validation" in result.claims[0].evidence.lower()


def test_validate_evidence_quotes_rejects_an_empty_quote():
    verification = judge.Verification(
        claims=[
            judge.ClaimCheck(
                claim="FastAPI supports feature X.",
                relevant=True,
                supported=True,
                evidence="No quotation was provided.",
                evidence_quote="",
            )
        ]
    )
    result = judge.validate_evidence_quotes(
        verification, "FastAPI supports dependency injection."
    )
    assert result.claims[0].supported is False
    assert result.claims[0].evidence_quote == ""


def test_validate_evidence_quotes_preserves_unsupported_claims():
    original_evidence = (
        "The supplied context does not establish this claim."
    )
    verification = judge.Verification(
        claims=[
            judge.ClaimCheck(
                claim="FastAPI supports feature X.",
                relevant=True,
                supported=False,
                evidence=original_evidence,
                evidence_quote="",
            )
        ]
    )
    result = judge.validate_evidence_quotes(
        verification, "FastAPI supports dependency injection."
    )
    assert result.claims[0].supported is False
    assert result.claims[0].evidence == original_evidence


def test_validate_evidence_quotes_accepts_equivalent_decorator_quote_styles():
    verification = judge.Verification(
        claims=[
            judge.ClaimCheck(
                claim="Define a GET endpoint using @app.get('/').",
                relevant=True,
                supported=True,
                evidence="The context explains how to define a GET endpoint.",
                evidence_quote=(
                    "You can use the @app.get('/') decorator "
                    "to define a GET endpoint."
                ),
            )
        ]
    )
    context = (
        'You can use the @app.get("/") decorator to define a GET endpoint.'
    )

    result = judge.validate_evidence_quotes(verification, context)

    assert result.claims[0].supported is True
    assert result.claims[0].evidence_quote == (
        "You can use the @app.get('/') decorator "
        "to define a GET endpoint."
    )

def test_validate_evidence_quotes_accepts_punctuation_spacing_differences():
    verification = judge.Verification(
        claims=[
            judge.ClaimCheck(
                claim="Run the development server.",
                relevant=True,
                supported=True,
                evidence="The context describes the server command.",
                evidence_quote="Run the development server,use fastapi dev.",
            )
        ]
    )
    context = "Run the development server, use fastapi dev."

    result = judge.validate_evidence_quotes(verification, context)

    assert result.claims[0].supported is True
    assert result.claims[0].evidence_quote == (
        "Run the development server,use fastapi dev."
    )



def test_verify_claims_rejects_claim_when_entailment_check_fails(monkeypatch):
    verification = judge.Verification(
        claims=[
            judge.ClaimCheck(
                claim="The development server is started with `fastapi dev`.",
                relevant=True,
                supported=True,
                evidence="The heading indicates that the server can be run.",
                evidence_quote="Run the live server:",
            )
        ]
    )
    monkeypatch.setattr(judge, "chat_json", Mock(return_value=verification))
    entailment = Mock(
        return_value=judge.EntailmentCheck(
            supported=False,
            explanation="The evidence does not specify the command.",
        )
    )
    monkeypatch.setattr(judge, "verify_entailment", entailment)

    result = judge.verify_claims(
        question="How do I start the development server?",
        answer="The development server is started with `fastapi dev`.",
        context="Run the live server:",
    )

    claim = result.claims[0]
    assert claim.supported is False
    assert claim.evidence_quote == ""
    assert "second-pass entailment check" in claim.evidence.lower()
    entailment.assert_called_once_with(
        claim="The development server is started with `fastapi dev`.",
        evidence_quote="Run the live server:",
    )


def test_verify_claims_preserves_claim_accepted_by_entailment_check(monkeypatch):
    quote = "The development server can be started using `fastapi dev`."
    verification = judge.Verification(
        claims=[
            judge.ClaimCheck(
                claim="The development server is started with `fastapi dev`.",
                relevant=True,
                supported=True,
                evidence="The context specifies the development command.",
                evidence_quote=quote,
            )
        ]
    )
    monkeypatch.setattr(judge, "chat_json", Mock(return_value=verification))
    monkeypatch.setattr(
        judge,
        "verify_entailment",
        Mock(
            return_value=judge.EntailmentCheck(
                supported=True,
                explanation="The evidence directly specifies the command.",
            )
        ),
    )

    result = judge.verify_claims(
        question="How do I start the development server?",
        answer="The development server is started with `fastapi dev`.",
        context=quote,
    )

    assert result.claims[0].supported is True
    assert result.claims[0].evidence_quote == quote

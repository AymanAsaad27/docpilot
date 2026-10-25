from unittest import result

from eval.run_evaluation import (
    citation_metrics,
    source_metrics,
    appears_to_abstain,
    summarize,
)


def test_source_metrics_detects_matching_source():
    passages = [
        {"source": "tutorial/first-steps.md"},
        {"source": "tutorial/query-params.md"},
    ]

    result = source_metrics(
        ["tutorial/first-steps.md", "tutorial/body.md"],
        passages,
    )

    assert result["matched_sources"] == ["tutorial/first-steps.md"]
    assert result["source_recall"] == 0.5
    assert result["any_expected_source_found"] is True


def test_source_metrics_when_nothing_matches():
    result = source_metrics(
        ["tutorial/body.md"],
        [{"source": "tutorial/first-steps.md"}],
    )

    assert result["source_recall"] == 0.0
    assert result["any_expected_source_found"] is False


def test_citation_metrics_detects_invalid_marker():
    passages = [{"citation": 1}, {"citation": 2}]

    result = citation_metrics("See [1] and [3].", passages)

    assert result["citation_count"] == 2
    assert result["has_citations"] is True
    assert result["invalid_citations"] == [3]
    assert result["citation_valid"] is False


def test_citation_metrics_accepts_valid_markers():
    passages = [{"citation": 1}, {"citation": 2}]

    result = citation_metrics("See [1] and [2].", passages)

    assert result["citation_valid"] is True
    assert result["invalid_citations"] == []


def test_abstention_detection():
    assert appears_to_abstain(
        "The documentation does not specify this behavior."
    )
    assert not appears_to_abstain(
        "Use a Pydantic model for the request body."
    )


def test_summary_handles_empty_results():
    result = summarize([])

    assert result["total_questions"] == 0
    assert result["successful_runs"] == 0
    assert result["failed_runs"] == 0

def test_summary_reports_valid_citation_coverage():
    base = {
        "error": None,
        "source_recall": 1.0,
        "any_expected_source_found": True,
        "appears_to_abstain": False,
        "abstention_match": True,
        "faithfulness": 1.0,
        "relevance": 1.0,
        "correctness": 1.0,
        "latency_seconds": 1.0,
        "retries": 0,
        "regenerations": 0,
    }

    records = [
        {
            **base,
            "expected_answerable": True,
            "has_citations": True,
            "citation_valid": True,
        },
        {
            **base,
            "expected_answerable": True,
            "has_citations": True,
            "citation_valid": False,
        },
        {
            **base,
            "expected_answerable": True,
            "has_citations": False,
            "citation_valid": True,
        },
        {
            **base,
            "expected_answerable": False,
            "has_citations": False,
            "citation_valid": True,
            "appears_to_abstain": True,
        },
    ]

    result = summarize(records)

    assert result["answerable_questions_with_citations"] == 0.6667
    assert (
        result["answerable_questions_with_valid_citations"]
        == 0.3333
)

def test_abstention_detects_docpilot_insufficient_context_message():
    answer = (
        "I can't answer this reliably from the supplied documentation. "
        "The retrieved context does not contain enough information "
        "to establish the answer."
    )

    assert appears_to_abstain(answer)


def test_abstention_detects_missing_server_capacity():
    answer = (
        "The documentation does not contain enough information "
        "to establish the exact server capacity."
    )

    assert appears_to_abstain(answer)


def test_abstention_does_not_match_normal_technical_answer():
    answer = (
        "Use a Pydantic model to validate the request body, "
        "then return the result from your path operation."
    )

    assert not appears_to_abstain(answer)

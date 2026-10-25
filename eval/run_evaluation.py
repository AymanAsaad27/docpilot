import json
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import run_agent  # noqa: E402


DATASET = ROOT / "data" / "evaluation" / "questions.json"
OUTPUT_DIR = ROOT / "results" / "evaluation"
CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def citation_metrics(answer, passages):
    groups = CITATION_PATTERN.findall(answer)

    markers = [
        int(number.strip())
        for group in groups
        for number in group.split(",")
    ]

    available = {p["citation"] for p in passages}
    invalid = sorted(set(markers) - available)

    return {
        "citation_count": len(markers),
        "has_citations": bool(markers),
        "invalid_citations": invalid,
        "citation_valid": not invalid,
    }


def source_metrics(expected_sources, passages):
    retrieved = {
        str(p.get("source", "")).lstrip("./")
        for p in passages
    }
    expected = {str(s).lstrip("./") for s in expected_sources}
    matched = sorted(expected & retrieved)
    return {
        "expected_sources": sorted(expected),
        "retrieved_sources": sorted(retrieved),
        "matched_sources": matched,
        "source_recall": len(matched) / len(expected) if expected else None,
        "any_expected_source_found": bool(matched),
    }


def appears_to_abstain(answer):
    pattern = re.compile(
        r"\b("
        r"cannot determine|can't determine|"
        r"not enough information|insufficient information|"
        r"not provided in the documentation|"
        r"unable to establish|cannot establish|can't establish|"
        r"does not specify|not specified in the documentation|"
        r"documentation does not say|couldn't find|"
        r"can't answer this reliably|cannot answer this reliably|"
        r"unable to answer|"
        r"does not contain enough information|"
        r"not enough information to establish|"
        r"cannot be established from the documentation|"
        r"can't be established from the documentation"
        r")\b",
        re.IGNORECASE,
    )
    return bool(pattern.search(answer))


def safe_mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.mean(values), 4) if values else None


def evaluate_question(question_record):
    started = time.perf_counter()
    question = question_record["question"]

    try:
        result = run_agent(question)
        answer = result.get("answer", "")
        passages = result.get("citations", [])

        sources = source_metrics(
            question_record.get("relevant_sources", []),
            passages,
        )
        citations = citation_metrics(answer, passages)
        abstention = appears_to_abstain(answer)

        expected_answerable = question_record["answerable"]
        return {
            "id": question_record["id"],
            "category": question_record["category"],
            "question": question,
            "expected_answerable": expected_answerable,
            "expected_answer": question_record.get("expected_answer", ""),
            "answer": answer,
            "error": None,
            **sources,
            **citations,
            "appears_to_abstain": abstention,
            "abstention_expected": not expected_answerable,
            "abstention_match": (
                abstention == (not expected_answerable)
            ),
            "context_sufficient": result.get("context_sufficient"),
            "low_confidence": result.get("low_confidence"),
            "faithfulness": result.get("faithfulness"),
            "relevance": result.get("relevance"),
            "correctness": result.get("correctness"),
            "latency_seconds": result.get(
                "latency_seconds",
                round(time.perf_counter() - started, 3),
            ),
            "retries": result.get("retries", 0),
            "regenerations": result.get("regenerations", 0),
        }
    except Exception as exc:
        return {
            "id": question_record.get("id"),
            "category": question_record.get("category"),
            "question": question,
            "expected_answerable": question_record.get("answerable"),
            "answer": "",
            "error": f"{type(exc).__name__}: {exc}",
            "latency_seconds": round(time.perf_counter() - started, 3),
        }


def summarize(records):
    successful = [r for r in records if not r.get("error")]
    answerable = [r for r in successful if r["expected_answerable"]]
    unanswerable = [r for r in successful if not r["expected_answerable"]]

    return {
        "total_questions": len(records),
        "successful_runs": len(successful),
        "failed_runs": len(records) - len(successful),
        "source_recall": safe_mean(
            [r.get("source_recall") for r in answerable]
        ),
        "expected_source_hit_rate": safe_mean(
            [float(r["any_expected_source_found"]) for r in answerable]
        ),
                "citation_validity_rate": safe_mean(
            [float(r["citation_valid"]) for r in successful]
        ),
        "answerable_questions_with_citations": safe_mean(
            [float(r["has_citations"]) for r in answerable]
        ),
        "answerable_questions_with_valid_citations": safe_mean(
            [
                float(r["has_citations"] and r["citation_valid"])
                for r in answerable
            ]
        ),
        "unanswerable_abstention_rate": safe_mean(
            [float(r["appears_to_abstain"]) for r in unanswerable]
        ),
        "answerability_match_rate": safe_mean(
            [float(r["abstention_match"]) for r in successful]
        ),
        "mean_faithfulness": safe_mean(
            [r.get("faithfulness") for r in successful]
        ),
        "mean_relevance": safe_mean(
            [r.get("relevance") for r in successful]
        ),
        "mean_correctness": safe_mean(
            [r.get("correctness") for r in successful]
        ),
        "mean_latency_seconds": safe_mean(
            [r.get("latency_seconds") for r in successful]
        ),
        "mean_retries": safe_mean([r.get("retries") for r in successful]),
        "mean_regenerations": safe_mean(
            [r.get("regenerations") for r in successful]
        ),
    }


def write_markdown(report, path):
    summary = report["summary"]
    lines = [
        "# DocPilot Evaluation Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Overall results",
        "",
        "| Metric | Result |",
        "|---|---:|",
    ]

    for key, value in summary.items():
        lines.append(f"| {key.replace('_', ' ').title()} | {value} |")

    lines += [
        "",
        "## Per-question results",
        "",
        "| ID | Category | Source recall | Citation valid | Abstention match | Faithfulness | Correctness | Latency (s) | Error |",
        "|---|---|---:|---|---|---:|---:|---:|---|",
    ]

    for r in report["results"]:
        lines.append(
            f"| {r.get('id', '')} "
            f"| {r.get('category', '')} "
            f"| {r.get('source_recall', '')} "
            f"| {r.get('citation_valid', '')} "
            f"| {r.get('abstention_match', '')} "
            f"| {r.get('faithfulness', '')} "
            f"| {r.get('correctness', '')} "
            f"| {r.get('latency_seconds', '')} "
            f"| {(r.get('error') or '').replace('|', '/')} |"
        )

    lines += [
        "",
        "## Interpretation notes",
        "",
        "- Source recall measures expected source-file overlap among returned passages.",
        "- Citation validity checks whether numeric citation markers point to available passages; it does not prove the cited passage supports the claim.",
        "- Abstention is detected using a phrase-based heuristic and requires manual review.",
        "- Faithfulness, relevance, and correctness are model-judged scores, not independently verified ground truth.",
        "- Review individual answers and errors before treating these results as a quality claim.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    questions = json.loads(DATASET.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("Expected the evaluation dataset to be a JSON list.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"Evaluating {len(questions)} questions.")
    print("This calls your local models and may take some time.")

    for index, record in enumerate(questions, start=1):
        print(
            f"[{index}/{len(questions)}] "
            f"{record.get('id', '?')}: {record['question']}"
        )
        result = evaluate_question(record)
        results.append(result)

        if result.get("error"):
            print(f"  ERROR: {result['error']}")
        else:
            print(
                f"  source_recall={result.get('source_recall')} "
                f"faithfulness={result.get('faithfulness')} "
                f"latency={result.get('latency_seconds')}s"
            )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET.relative_to(ROOT)),
        "summary": summarize(results),
        "results": results,
    }

    json_path = OUTPUT_DIR / "report.json"
    markdown_path = OUTPUT_DIR / "report.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(report, markdown_path)

    print("\nEvaluation finished.")
    print(f"JSON report: {json_path.relative_to(ROOT)}")
    print(f"Markdown report: {markdown_path.relative_to(ROOT)}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
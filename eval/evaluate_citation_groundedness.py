import json
import re
from pathlib import Path

from src.agent import retrieve_for_queries
from src.judge import verify_cited_claims

REPORT_PATH = Path("results/evaluation/report.json")
TRACE_PATH = Path("logs/traces.jsonl")
OUTPUT_PATH = Path("results/evaluation/citation_groundedness_pilot.json")
PILOT_IDS = {"F001", "F002", "F003"}

CITATION_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def save_progress(data):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = OUTPUT_PATH.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(OUTPUT_PATH)


def extract_citation_ids(answer):
    ids = []
    for match in CITATION_PATTERN.finditer(answer or ""):
        ids.extend(
            int(value.strip())
            for value in match.group(1).split(",")
        )
    return sorted(set(ids))


def main():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    traces = {}

    if TRACE_PATH.exists():
        for line in TRACE_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            question = record.get("question")
            if question:
                # Keep the most recent trace for each question.
                traces[question] = record

    previous = {"results": []}
    if OUTPUT_PATH.exists():
        try:
            previous = json.loads(
                OUTPUT_PATH.read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError):
            pass

    completed = {
        item["id"]: item
        for item in previous.get("results", [])
        if item.get("status") == "complete"
    }

    data = {
        "evaluation": "citation_groundedness_pilot",
        "context_type": "reconstructed_not_original",
        "warning": (
            "Passages were retrieved again; original passage text and "
            "citation mapping were not saved in the benchmark report."
        ),
        "results": list(completed.values()),
    }
    save_progress(data)

    for item in report.get("results", []):
        question_id = item.get("id")
        if question_id not in PILOT_IDS or question_id in completed:
            continue

        question = item["question"]
        answer = item.get("answer") or ""
        trace = traces.get(question, {})
        queries = trace.get("queries") or [question]

        print(f"\nEvaluating {question_id}: {question}", flush=True)

        try:
            passages = retrieve_for_queries(queries)

            citations = []
            for index, passage in enumerate(passages, start=1):
                metadata = passage.get("metadata", {})
                citations.append({
                    "citation": index,
                    "source": metadata.get("source", "Unknown"),
                    "heading": metadata.get("heading", "Unknown"),
                    "rerank_score": passage.get("rerank_score"),
                    "text": passage.get("text", ""),
                })

            cited_ids = extract_citation_ids(answer)
            available_ids = {entry["citation"] for entry in citations}
            out_of_range = sorted(set(cited_ids) - available_ids)

            verification = verify_cited_claims(
                question=question,
                answer=answer,
                citations=citations,
            )
            claims = [
                claim.model_dump()
                for claim in verification.claims
            ]

            supported_count = sum(
                claim["supported"] for claim in claims
            )
            claim_count = len(claims)

            result = {
                "id": question_id,
                "question": question,
                "status": "complete",
                "context_type": "reconstructed_not_original",
                "queries_used": queries,
                "reconstructed_passage_count": len(citations),
                "cited_ids_in_answer": cited_ids,
                "citation_ids_out_of_range_in_reconstructed_context": (
                    out_of_range
                ),
                "cited_claim_count": claim_count,
                "supported_claim_count": supported_count,
                "unsupported_claim_count": claim_count - supported_count,
                "groundedness_rate": (
                    supported_count / claim_count
                    if claim_count else None
                ),
                "claims": claims,
            }

        except Exception as exc:
            result = {
                "id": question_id,
                "question": question,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

        data["results"] = [
            existing
            for existing in data["results"]
            if existing.get("id") != question_id
        ]
        data["results"].append(result)
        completed[question_id] = result
        save_progress(data)

        print(
            f"Saved {question_id}: status={result['status']}",
            flush=True,
        )
        if result["status"] == "complete":
            print(
                f"  Cited claims: {result['cited_claim_count']}; "
                f"supported: {result['supported_claim_count']}; "
                f"groundedness rate: {result['groundedness_rate']}",
                flush=True,
            )
        else:
            print(f"  Error: {result['error']}", flush=True)

    print(f"\nPilot results saved to: {OUTPUT_PATH}")
    print("Existing completed questions will be skipped on rerun.")


if __name__ == "__main__":
    main()

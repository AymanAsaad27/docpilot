
import json
import re
from pathlib import Path

from src.agent import retrieve_for_queries
from src.judge import verify_claims

REPORT_PATH = Path("results/evaluation/report.json")
TRACE_PATH = Path("logs/traces.jsonl")
OUTPUT_PATH = Path(
    "results/evaluation/answer_groundedness_pilot.json"
)
PILOT_IDS = {"F001", "F002", "F003"}


def save_progress(data):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT_PATH.with_suffix(".tmp")
    temp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp.replace(OUTPUT_PATH)


def remove_source_footer(answer):
    """Remove appended source-list lines, not factual answer text."""
    return re.sub(
        r"(?:\n\s*\[\d+\]\s+Source:.*)+\s*$",
        "",
        answer or "",
        flags=re.IGNORECASE,
    )


def main():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    traces = {}
    if TRACE_PATH.exists():
        for line in TRACE_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                question = record.get("question")
                if question:
                    traces[question] = record

    data = {
        "evaluation": "answer_level_groundedness_pilot",
        "context_type": "reconstructed_not_original",
        "metric_definition": (
            "Fraction of relevant factual claims judged supported "
            "by the reconstructed documentation context. Claims "
            "classified as irrelevant are excluded from the rate."
        ),
        "citation_attribution": (
            "Not evaluated: original citation-to-passage mapping "
            "was not saved."
        ),
        "results": [],
    }

    # Rebuild the pilot from scratch so old results without relevance
    # classifications cannot silently remain in the output.
    save_progress(data)

    for item in report.get("results", []):
        question_id = item.get("id")
        if question_id not in PILOT_IDS:
            continue

        question = item["question"]
        answer = remove_source_footer(item.get("answer") or "")
        queries = (
            traces.get(question, {}).get("queries") or [question]
        )

        print(f"\nEvaluating {question_id}: {question}", flush=True)

        try:
            passages = retrieve_for_queries(queries)

            context_blocks = []
            for index, passage in enumerate(passages, start=1):
                metadata = passage.get("metadata", {})
                context_blocks.append(
                    f"[{index}] Source: "
                    f"{metadata.get('source', 'Unknown')}\n"
                    f"Section: "
                    f"{metadata.get('heading', 'Unknown')}\n"
                    f"{passage.get('text', '')}"
                )

            context = "\n\n---\n\n".join(context_blocks)

            verification = verify_claims(
                question=question,
                answer=answer,
                context=context,
            )

            claims = [
                claim.model_dump()
                for claim in verification.claims
            ]

            relevant_claims = [
                claim for claim in claims
                if claim["relevant"]
            ]
            irrelevant_claims = [
                claim for claim in claims
                if not claim["relevant"]
            ]

            supported_relevant = sum(
                claim["supported"] for claim in relevant_claims
            )
            unsupported_relevant = (
                len(relevant_claims) - supported_relevant
            )

            supported_irrelevant = sum(
                claim["supported"] for claim in irrelevant_claims
            )
            relevant_with_verified_quotes = sum(
                claim["supported"]
                and bool(claim.get("evidence_quote", "").strip())
                for claim in relevant_claims
            )

            relevant_without_verified_quotes = (
                len(relevant_claims) - relevant_with_verified_quotes
            )
            result = {
                "id": question_id,
                "question": question,
                "status": "complete",
                "context_type": "reconstructed_not_original",
                "queries_used": queries,
                "reconstructed_passage_count": len(passages),
                "claim_count": len(claims),
                "relevant_claim_count": len(relevant_claims),
                "relevant_claims_with_verified_quotes": relevant_with_verified_quotes,
                "relevant_claims_without_verified_quotes": relevant_without_verified_quotes,
                "irrelevant_claim_count": len(irrelevant_claims),
                "supported_relevant_claim_count": supported_relevant,
                "unsupported_relevant_claim_count": unsupported_relevant,
                "supported_irrelevant_claim_count": supported_irrelevant,
                "groundedness_rate": (
                    supported_relevant / len(relevant_claims)
                    if relevant_claims else None
                ),
                "claims": claims,
                "citation_attribution": (
                    "Not evaluated; original citation mapping unavailable."
                ),
            }

        except Exception as exc:
            result = {
                "id": question_id,
                "question": question,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

        data["results"].append(result)
        save_progress(data)

        print(f"Saved {question_id}: {result['status']}", flush=True)

        if result["status"] == "complete":
            print(
                f"  Total claims: {result['claim_count']}; "
                f"relevant: {result['relevant_claim_count']}; "
                f"irrelevant: {result['irrelevant_claim_count']}",
                flush=True,
            )
            print(
                f"  Supported relevant: "
                f"{result['supported_relevant_claim_count']}; "
                f"unsupported relevant: "
                f"{result['unsupported_relevant_claim_count']}",
                flush=True,
            )
            print(
                f"  Groundedness: {result['groundedness_rate']}",
                flush=True,
            )
        else:
            print(f"  Error: {result['error']}", flush=True)

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

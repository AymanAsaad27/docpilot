import json
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from src.config import (
    FAITHFULNESS_MIN,
    LOG_FILE,
    MAX_REGENERATIONS,
    MAX_RETRIES,
    TOP_K_FINAL,
)
from src.judge import (
    faithfulness_score,
    grade_context,
    score_answer,
    verify_claims,
)
from src.llm import chat, chat_json
from src.retrieval import search


class Queries(BaseModel):
    queries: list[str] = Field(
        description="Alternative search queries that preserve the user's intent."
    )


class GeneratedAnswer(BaseModel):
    answer: str = Field(
        description="A direct answer supported by the supplied documentation."
    )


def rewrite_queries(question: str) -> list[str]:
    """Generate alternative search queries for a user question."""
    prompt = f"""
Generate exactly two alternative search queries for the question below.
Preserve its intent. Use different wording or relevant technical terms.
Return only the structured output.

Question: {question}
"""
    result = chat_json(
        prompt=prompt,
        schema=Queries,
        system="You are a search-query rewriting assistant.",
    )

    alternatives = []
    for query in result.queries:
        query = query.strip()
        if query and query.casefold() != question.strip().casefold():
            if query.casefold() not in {
                item.casefold() for item in alternatives
            }:
                alternatives.append(query)

    return alternatives[:2]


def retrieve_for_queries(queries: list[str]) -> list[dict]:
    """Search all queries and deduplicate results by chunk ID."""
    unique = {}

    for query in queries:
        for result in search(query):
            chunk_id = result["id"]
            existing = unique.get(chunk_id)

            if (
                existing is None
                or result["rerank_score"] > existing["rerank_score"]
            ):
                unique[chunk_id] = result

    ranked = sorted(
        unique.values(),
        key=lambda item: item["rerank_score"],
        reverse=True,
    )
    return ranked[:TOP_K_FINAL]


def format_context(passages: list[dict]) -> str:
    """Format passages with stable numbered source references."""
    blocks = []

    for index, passage in enumerate(passages, start=1):
        metadata = passage["metadata"]
        blocks.append(
            f"[{index}] Source: {metadata['source']}\n"
            f"Section: {metadata.get('heading', 'Unknown')}\n"
            f"{passage['text']}"
        )

    return "\n\n---\n\n".join(blocks)


def generate_answer(
    question: str,
    context: str,
    previous_answer: str | None = None,
    verification_feedback: str | None = None,
) -> str:
    """Generate a cited answer grounded in retrieved documentation."""
    retry_instructions = ""

    if previous_answer and verification_feedback:
        retry_instructions = f"""
Your previous answer did not pass verification.
Review this feedback and correct the answer:
{verification_feedback}

Previous answer:
{previous_answer}
"""

    prompt = f"""
Answer the user's question using only the documentation context below.

Follow these requirements:
1. Answer the question directly and completely.
2. Include a short, runnable Python example when the context supports one.
3. Explain briefly what the example does.
4. Cite factual statements using source markers such as [1] or [2].
5. Only use citation numbers that appear in the supplied context.
6. Do not invent missing details or rely on outside knowledge.
7. If the context is insufficient, explain what cannot be established.
8. Return a complete answer. Do not stop after an introductory sentence.

Question:
{question}

Documentation context:
{context}

{retry_instructions}
"""

    result = chat(
        prompt=prompt,
        system=(
            "You are DocPilot, a documentation assistant. "
            "Ground answers in the supplied sources and cite factual claims. "
            "Return a complete answer, including code examples when appropriate."
        ),
        temperature=0.0,
    )
    return result.strip()


def build_verification_feedback(verification) -> str:
    """Summarize unsupported claims for a possible regeneration."""
    unsupported = [
        claim for claim in verification.claims if not claim.supported
    ]

    if not unsupported:
        return "No unsupported claims were identified."

    return "\n".join(
        f"- Unsupported claim: {claim.claim}. Evidence/reason: {claim.evidence}"
        for claim in unsupported
    )


def log_trace(trace: dict) -> None:
    """Append one JSON trace record to the project log."""
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(trace, ensure_ascii=False) + "\n")


def run_agent(question: str) -> dict:
    """Run DocPilot's retrieval, answer, verification, and logging workflow."""
    started = time.perf_counter()
    question = question.strip()

    if not question:
        raise ValueError("Question must not be empty.")

    trace = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "events": [],
        "retries": 0,
        "regenerations": 0,
    }

    try:
        alternatives = rewrite_queries(question)
        queries = [question, *alternatives]
        trace["queries"] = queries
        trace["events"].append({
            "stage": "query_rewrite",
            "alternative_count": len(alternatives),
        })

        passages = retrieve_for_queries(queries)
        trace["events"].append({
            "stage": "retrieval",
            "passage_count": len(passages),
        })

        sufficient = False
        sufficiency_reason = "No passages retrieved."

        for attempt in range(MAX_RETRIES + 1):
            context = format_context(passages)

            if not context:
                sufficient = False
                sufficiency_reason = (
                    "No documentation passages were retrieved."
                )
            else:
                grade = grade_context(question, context)
                sufficient = grade.sufficient
                sufficiency_reason = grade.reason

            trace["events"].append({
                "stage": "context_grading",
                "attempt": attempt,
                "sufficient": sufficient,
                "reason": sufficiency_reason,
            })

            if sufficient or attempt >= MAX_RETRIES:
                break

            trace["retries"] += 1

            retry_query = (
                f"{question}\n"
                f"Find documentation that addresses this missing information: "
                f"{sufficiency_reason}"
            )
            retry_passages = search(retry_query)

            combined = {p["id"]: p for p in passages}
            for passage in retry_passages:
                combined[passage["id"]] = passage

            passages = sorted(
                combined.values(),
                key=lambda item: item["rerank_score"],
                reverse=True,
            )[:TOP_K_FINAL]

        context = format_context(passages)

        # IMPORTANT: Do not generate an answer when the context
        # sufficiency judge says the documentation is insufficient.
        if not sufficient:
            answer = (
                "I can't answer this reliably from the supplied documentation. "
                "The retrieved context does not contain enough information "
                "to establish the answer."
            )

            trace["events"].append({
                "stage": "abstention",
                "reason": sufficiency_reason,
            })
        else:
            answer = generate_answer(question, context)

        # Verify the response, but never regenerate an abstention
        # into a speculative answer.
        verification = verify_claims(question, answer, context)
        faithfulness = faithfulness_score(verification)

        trace["events"].append({
            "stage": "verification",
            "faithfulness": faithfulness,
            "claim_count": len(verification.claims),
            "unsupported_claim_count": sum(
                not claim.supported for claim in verification.claims
            ),
        })

        if sufficient:
            for _ in range(MAX_REGENERATIONS):
                if faithfulness >= FAITHFULNESS_MIN:
                    break

                feedback = build_verification_feedback(verification)
                answer = generate_answer(
                    question,
                    context,
                    previous_answer=answer,
                    verification_feedback=feedback,
                )
                verification = verify_claims(question, answer, context)
                faithfulness = faithfulness_score(verification)
                trace["regenerations"] += 1

        scores = score_answer(question, answer, context)

        citations = []
        for index, passage in enumerate(passages, start=1):
            citations.append({
                "citation": index,
                "source": passage["metadata"]["source"],
                "heading": passage["metadata"].get("heading", "Unknown"),
                "rerank_score": passage.get("rerank_score"),
                "text": passage["text"],
            })

        low_confidence = (
            not sufficient
            or faithfulness < FAITHFULNESS_MIN
            or scores.correctness < 4
        )

        trace["faithfulness"] = faithfulness
        trace["relevance"] = scores.relevance / 5
        trace["correctness"] = scores.correctness / 5
        trace["low_confidence"] = low_confidence
        trace["latency_seconds"] = round(
            time.perf_counter() - started, 3
        )

        result = {
            "question": question,
            "answer": answer,
            "citations": citations,
            "faithfulness": faithfulness,
            "relevance": scores.relevance / 5,
            "correctness": scores.correctness / 5,
            "low_confidence": low_confidence,
            "context_sufficient": sufficient,
            "context_reason": sufficiency_reason,
            "latency_seconds": trace["latency_seconds"],
            "retries": trace["retries"],
            "regenerations": trace["regenerations"],
            "trace": trace,
        }

        trace["events"].append({"stage": "completed"})
        log_trace(trace)
        return result

    except Exception as exc:
        trace["events"].append({
            "stage": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        trace["latency_seconds"] = round(
            time.perf_counter() - started, 3
        )
        log_trace(trace)
        raise
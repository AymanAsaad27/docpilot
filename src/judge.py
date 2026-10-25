import re
from pydantic import BaseModel, Field

from src.config import JUDGE_MODEL
from src.llm import chat_json


class Grade(BaseModel):
    sufficient: bool = Field(
        description="Whether the supplied context is sufficient to answer the question."
    )
    reason: str = Field(
        description="Brief explanation of the sufficiency decision."
    )


class ClaimCheck(BaseModel):
    claim: str = Field(description="One factual claim from the answer.")
    relevant: bool = Field(
        description="Whether the claim helps answer the user's question."
    )
    supported: bool = Field(
        description="Whether the supplied evidence directly supports the claim."
    )
    evidence: str = Field(
        description="Brief explanation of why the claim is or is not supported."
    )
    evidence_quote: str = Field(
    description=(
        "For a supported claim, provide an exact, verbatim quotation from "
        "the supplied context. For an unsupported claim, use an empty string."
    ),
)
class Verification(BaseModel):
    claims: list[ClaimCheck] = Field(
        description="Factual claims checked against the supplied context."
    )

class EntailmentCheck(BaseModel):
    supported: bool = Field(
        description=(
            "Whether the quoted evidence directly supports every material "
            "part of the claim."
        )
    )
    explanation: str = Field(
        description="Brief explanation of why the evidence does or does not support the claim."
    )

def validate_evidence_quotes(
    verification: Verification,
    context: str,
) -> Verification:
    """Reject supported claims whose evidence quotation is unverifiable."""

    def normalize(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()

        # Normalize straight and curly quote styles.
        text = (
            text.replace("‘", "'")
            .replace("’", "'")
            .replace("“", '"')
            .replace("”", '"')
        )

        # Ignore whitespace around punctuation for robust exact-text matching.
        text = re.sub(r"\s*([,.;:!?])\s*", r"\1", text)

        return text

    def normalize_decorator_quotes(text: str) -> str:
        # Treat equivalent quote styles around decorator arguments as equal.
        text = re.sub(
            r"""(@[\w.]+\s*\(\s*)['"]([^'"]*)['"](\s*\))""",
            r'\1"\2"\3',
            text,
        )
        return text

    normalized_context = normalize_decorator_quotes(normalize(context))

    for claim in verification.claims:
        if not claim.supported:
            continue

        quote = normalize_decorator_quotes(normalize(claim.evidence_quote))

        if not quote or quote not in normalized_context:
            claim.supported = False
            claim.evidence = (
                "Marked unsupported by deterministic validation: "
                "the supporting quotation is empty or does not appear "
                "in the supplied context."
            )
            claim.evidence_quote = ""

    return verification




class Score(BaseModel):
    relevance: int = Field(
        ge=1,
        le=5,
        description="How relevant the answer is to the question, from 1 to 5.",
    )
    correctness: int = Field(
        ge=1,
        le=5,
        description="How correct the answer is based on the context, from 1 to 5.",
    )
    explanation: str = Field(
        description="Brief explanation of the ratings."
    )


def grade_context(question: str, context: str) -> Grade:
    """Judge whether the retrieved passages contain enough information."""
    prompt = f"""
Decide whether the context contains enough reliable information to answer
the user's question. Do not use outside knowledge to fill gaps.

Question:
{question}

Context:
{context}
"""
    return chat_json(
        prompt=prompt,
        schema=Grade,
        system=(
            "You are a careful retrieval-quality evaluator. "
            "Return valid structured output."
        ),
        model=JUDGE_MODEL,
    )


def verify_entailment(
    claim: str,
    evidence_quote: str,
) -> EntailmentCheck:
    """Check whether a quoted passage entails the full claim."""
    prompt = f"""
Decide whether the EVIDENCE directly supports the entire CLAIM.

Rules:
- Judge only what the evidence explicitly establishes.
- Related wording or topical relevance is not sufficient.
- A heading or instruction that merely introduces information does not
  establish a specific factual claim.
- Check the subject, action, and every material detail of the claim.
- Do not use outside knowledge or infer missing facts.
- If the evidence does not establish the full claim, return supported=false.
- Explain the decision briefly.

CLAIM:
{claim}

EVIDENCE:
{evidence_quote}
"""
    return chat_json(
        prompt=prompt,
        schema=EntailmentCheck,
        system=(
            "You are a strict textual entailment auditor. "
            "Judge only whether the supplied evidence establishes the full "
            "claim. Do not assume missing information. Return valid structured output."
        ),
        model=JUDGE_MODEL,
    )




def verify_evidence(verification: Verification, context: str) -> Verification:
    """Apply deterministic quote validation and second-pass entailment checks."""
    verification = validate_evidence_quotes(verification, context)

    for claim in verification.claims:
        if not claim.supported:
            continue

        entailment = verify_entailment(
            claim=claim.claim,
            evidence_quote=claim.evidence_quote,
        )

        if not entailment.supported:
            claim.supported = False
            claim.evidence = (
                "Rejected by the second-pass entailment check: "
                f"{entailment.explanation}"
            )
            claim.evidence_quote = ""

    return verification


def verify_claims(question: str, answer: str, context: str) -> Verification:
    """Evaluate answer claims for relevance and contextual support."""
    prompt = f"""
You are auditing whether an answer is grounded in supplied documentation.

Evaluate claims stated in the ANSWER using the QUESTION and CONTEXT.

Claim extraction rules:
- Extract at most 12 distinct factual claims explicitly made in the answer.
- Each claim must be one complete, independently checkable proposition.
- Keep claims under 30 words.
- Do not split one proposition into overlapping or repetitive claims.
- Do not duplicate the same fact when the answer repeats it.
- Do not produce fragments, unfinished sentences, or claims ending mid-quotation.
- Do not extract claims that appear only in the context.
- Ignore headings, instructions, code syntax alone, and non-factual commentary.
- If no factual claims can be extracted, return an empty claims list.

Relevance rules:
- Set relevant=true when the claim helps answer the user's question.
- Set relevant=false when the claim is unrelated or merely incidental.
- Assess support independently of relevance.
- Judge relevance independently of whether the claim is supported.

Evidence rules:
- Judge support using ONLY the supplied CONTEXT.
- Evaluate each extracted claim independently.
- Decide whether the context logically supports the FULL claim before
  selecting a quotation. The presence of related words or a relevant
  sentence is not sufficient evidence of support.
- Apply this test: if a careful reader saw only the quoted passage,
  would they have enough evidence to accept the claim as stated?
- A quotation that merely introduces a topic, gives an instruction,
  names a heading, or appears near the relevant information does not
  establish a claim unless its content directly supports that claim.
- Do not treat a claim as supported merely because its wording resembles
  the context or because a relevant quotation exists.
- Check every material part of a compound claim, including its subject,
  action, conditions, quantities, comparisons, and qualifiers.
- If any material part is unsupported, set supported=false and explain
  what is missing. Do not mark a partially supported claim as supported.
- Only after deciding that the FULL claim is supported, select the exact
  sentence or continuous phrase in CONTEXT that establishes it.
- Every supported claim MUST have a nonempty evidence_quote copied
  directly from CONTEXT. Preserve the original wording and punctuation.
- Keep evidence_quote to the shortest complete sentence or continuous
  phrase that directly establishes the claim.
- Do not paraphrase, combine separate quotations, or create a quotation
  that does not appear in CONTEXT.
- Do not use the ANSWER itself as evidence that a claim is true.
- If no direct supporting quotation exists, set supported=false and
  evidence_quote="".
- Do not use outside knowledge or infer details not established by
  the context.
- For supported claims, explain briefly how the quoted text supports
  the full claim. For unsupported claims, explain what evidence is missing.
- Keep each evidence explanation under 40 words.

Return valid structured output matching the required schema.

QUESTION:
{question}

ANSWER TO AUDIT:
{answer}

DOCUMENTATION CONTEXT — THE ONLY SOURCE OF EVIDENCE:
{context}
"""

    verification = chat_json(
        prompt=prompt,
        schema=Verification,
        system=(
            "You are a strict documentation-groundedness auditor. "
            "Evaluate claims against the supplied documentation context. "
            "Assess relevance separately from support. For every supported claim, "
            "provide an exact, verbatim quotation from the context that directly "
            "supports it. Never invent evidence. Return valid structured output."
        ),
        model=JUDGE_MODEL,
    )


    return verify_evidence(verification, context)


def verify_cited_claims(
    question: str,
    answer: str,
    citations: list[dict],
) -> Verification:
    """Evaluate whether cited passages support the claims they accompany."""
    citation_context = "\n\n---\n\n".join(
        f"[{passage['citation']}] Source: {passage.get('source', 'Unknown')}\n"
        f"Section: {passage.get('heading', 'Unknown')}\n"
        f"{passage.get('text', '')}"
        for passage in citations
    )

    prompt = f"""
Evaluate whether the answer's citations support the factual claims they accompany.

Rules:
- Extract only factual claims with citation markers such as [1] or [2].
- Extract at most 12 claims, each under 30 words.
- Set relevant=true if the claim helps answer the question; otherwise false.
- Inspect only the passage or passages identified by each claim's citation markers.
- Set supported=true only if those cited passages directly support the claim.
- Do not use other passages to rescue an unsupported citation.
- Do not use outside knowledge as evidence.
- Keep each evidence explanation under 40 words.
- Ignore uncited claims.
- If no cited factual claims exist, return an empty claims list.
- Return valid structured output matching the required schema.

Question:
{question}

Answer:
{answer}

Available citation passages:
{citation_context}
"""
    verification = chat_json(
        prompt=prompt,
        schema=Verification,
        system=(
            "You are a strict citation-groundedness evaluator. Evaluate each claim "
            "against its cited passages only. Classify relevance separately from "
            "citation support and return valid structured output."
        ),
        model=JUDGE_MODEL,
    )
    return verify_evidence(verification, citation_context)


def score_answer(question: str, answer: str, context: str) -> Score:
    """Rate answer relevance and context-based correctness from 1 to 5."""
    prompt = f"""
Rate the answer using the question and supplied context.

Relevance:
1 = unrelated; 2 = mostly unrelated; 3 = partially answers the question;
4 = answers the question well; 5 = directly and completely answers it.

Correctness:
1 = mostly unsupported or contradicted; 2 = major unsupported claims;
3 = mixed support; 4 = largely supported; 5 = fully supported by the context.

Do not reward plausible statements unless the context supports them.

Question:
{question}

Answer:
{answer}

Context:
{context}
"""
    return chat_json(
        prompt=prompt,
        schema=Score,
        system=(
            "You are a strict answer-quality evaluator. "
            "Return valid structured output."
        ),
        model=JUDGE_MODEL,
    )


def faithfulness_score(verification: Verification) -> float:
    """Return the supported fraction among relevant factual claims."""
    relevant_claims = [
        claim for claim in verification.claims if claim.relevant
    ]
    if not relevant_claims:
        return 1.0

    return sum(claim.supported for claim in relevant_claims) / len(relevant_claims)

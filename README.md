# DocPilot — Local Agentic RAG for Documentation QA

DocPilot is a local-first documentation question-answering agent built with Python, Ollama, ChromaDB, and a cross-encoder reranker. It retrieves technical documentation, generates answers with source citations, evaluates evidence support, and can retry or regenerate answers when verification indicates low confidence.

The included corpus contains FastAPI documentation. The project is designed as an experiment in retrieval quality, grounded generation, citation verification, and evaluation—not as a production-ready service.

## Highlights

- **Local LLM inference:** Ollama models for answer generation, query rewriting, and evaluation.
- **Retrieval pipeline:** ChromaDB vector search, embedding-based retrieval, and cross-encoder reranking.
- **Agent workflow:** alternative-query retrieval, context sufficiency checks, answer generation, claim verification, and bounded retries/regeneration.
- **Evidence checks:** deterministic validation of evidence quotations followed by a second-pass entailment check.
- **Evaluation:** a 30-question dataset with answerable and unanswerable questions, source-recall measurements, citation checks, model-judged scores, and latency tracking.
- **Automated tests:** the latest recorded full test run passed 91 tests.

## Architecture

```text
User question
     |
     v
Query rewriting
     |
     v
Vector retrieval (ChromaDB)
     |
     v
Cross-encoder reranking
     |
     v
Top-k documentation passages
     |
     v
Context sufficiency check
     |
     v
Answer generation (Ollama)
     |
     v
Claim and evidence verification
     |
     +---- insufficient confidence / failed checks
     |                 |
     |                 v
     |         bounded retry or regeneration
     |                 |
     +-----------------+
     |
     v
Answer, citations, scores, and trace
```

## Technology

- Python
- Ollama
- ChromaDB
- Sentence Transformers
- Pydantic
- pytest

Default models are configured in `src/config.py`:

| Purpose | Default model |
|---|---|
| Answer generation | `qwen2.5:7b` |
| Claim and answer judging | `llama3.1:8b` |
| Embeddings | `qwen3-embedding:0.6b` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

These defaults are configurable. Model availability, disk space, and inference speed depend on your machine.

## Requirements

- Python 3.12 is the development environment used for the current test results.
- [Ollama](https://ollama.com/) installed and running.
- Sufficient memory and disk space for the selected models.
- Internet access for initial package, model, and reranker downloads.

## Setup

Clone the repository, then enter the project directory:

```bash
git clone <YOUR_REPOSITORY_URL>
cd docpilot
```

Create and activate a virtual environment on macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start Ollama and download the configured language and embedding models:

```bash
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull qwen3-embedding:0.6b
```

Ensure Ollama is available before running the application. The cross-encoder may also be downloaded from Hugging Face on first use.

## Index the documentation

The repository includes documentation under `data/fastapi_docs/`.

Build the index:

```bash
PYTHONPATH=. python -m src.ingest
```

To rebuild the existing collection:

```bash
PYTHONPATH=. python -m src.ingest --reset
```

**Warning:** `--reset` rebuilds the existing ChromaDB collection.

## Ask a question

The agent can be called from Python:

```python
from src.agent import run_agent

result = run_agent("How do I define a request body in FastAPI?")

print(result["answer"])
print(result["citations"])
```

Run this from the repository root with the virtual environment active and Ollama available.

The returned result also contains evaluation information such as faithfulness, relevance, correctness, confidence-related state, latency, retries, regenerations, and a trace. See `src/agent.py` for the exact result structure.

## Tests

Run the full test suite:

```bash
PYTHONPATH=. pytest -q
```

Latest recorded result:

```text
91 passed
```

This result reflects the test run recorded during development. Run the suite again after making changes to reproduce it.

## Evaluation

The saved benchmark report is available at:

- `results/evaluation/report.md`
- `results/evaluation/report.json`

The report covers 30 questions across documentation topics, including two unanswerable questions.

### Saved benchmark results

| Metric | Result |
|---|---:|
| Questions | 30 |
| Successful runs | 30/30 |
| Failed runs | 0 |
| Source recall | 94.64% |
| Expected-source hit rate | 100% |
| Numeric citation validity | 100% |
| Answerable questions with citations | 85.71% |
| Unanswerable abstention rate | 100% |
| Mean model-judged faithfulness | 0.9788 |
| Mean model-judged relevance | 0.9467 |
| Mean model-judged correctness | 0.9733 |
| Mean latency | 102.94 seconds |

These figures are from the **saved benchmark generated on October 23, 2026**. They have not been recomputed after subsequent judge changes.

To run the evaluation again:

```bash
PYTHONPATH=. python -m eval.run_evaluation
```

This invokes local models for each question and can take a substantial amount of time.

### How to interpret these metrics

- **Source recall** measures overlap between expected source files and retrieved passages.
- **Numeric citation validity** checks whether citation markers refer to available passages. It does not establish that the cited text semantically supports the associated claim.
- **Abstention detection** uses a phrase-based heuristic and can require manual review.
- **Faithfulness, relevance, and correctness** are model-judged scores, not independently verified ground truth.
- The separate groundedness pilot reports use reconstructed retrieval context rather than the exact original context from each historical run. They should be interpreted as exploratory diagnostics, not definitive benchmark scores.

A high model-judged score should not be interpreted as proof that every answer is correct or fully grounded.

## Project structure

```text
docpilot/
├── data/
│   └── fastapi_docs/        # Documentation corpus
├── eval/
│   ├── run_evaluation.py
│   ├── evaluate_answer_groundedness.py
│   └── evaluate_citation_groundedness.py
├── results/
│   └── evaluation/          # Saved reports and pilot results
├── src/
│   ├── agent.py              # Agent workflow
│   ├── config.py             # Models and configuration
│   ├── ingest.py             # Document indexing
│   ├── judge.py              # Grounding and answer evaluation
│   ├── llm.py                # Ollama integration
│   └── retrieval.py           # Retrieval and reranking
├── tests/
├── requirements.txt
└── README.md
```

## Current limitations

- Local inference has relatively high latency on the tested configuration.
- Model-based judges can make mistakes, including incorrectly accepting unsupported claims.
- Exact quotation validation establishes that a quote appears in the context, not that it logically entails the claim.
- Second-pass entailment checking is itself model-based and is not a formal guarantee.
- Citation validity and semantic citation support are different properties. Per-claim mapping to the exact passages referenced by each citation remains an important area for further validation.
- Historical traces do not preserve all original retrieved passage text and citation mappings, limiting exact replay of previous runs.
- The saved benchmark predates the latest judge changes.
- The project has not been presented as production-hardened or independently validated on a broad external benchmark.

## Future improvements

- Persist the exact retrieved passages and citation mappings in every trace.
- Strengthen per-claim citation attribution and regression coverage.
- Compare judge decisions against a manually annotated evaluation set.
- Improve latency and measure quality-versus-cost trade-offs.
- Add reproducible environment and model-version metadata to benchmark reports.

## License and attribution

No license has been declared for this repository yet. Review the licensing and attribution requirements for the included documentation and dependencies before publishing or redistributing the project.

## Documentation source and attribution

This project uses FastAPI documentation as its retrieval corpus:
- Documentation: https://fastapi.tiangolo.com/
- Source repository: https://github.com/fastapi/fastapi

The documentation corpus in `data/fastapi_docs/` is included for retrieval and evaluation in this portfolio project. FastAPI and its documentation are maintained by their respective contributors. Please consult the upstream repository for applicable licensing, attribution requirements, and current source material before redistributing the corpus.

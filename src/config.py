
from pathlib import Path

# Project root directory
ROOT = Path(__file__).resolve().parent.parent

# Data and storage paths
DOCS_DIR = ROOT / "data" / "fastapi_docs"
CHROMA_DIR = ROOT / "chroma_db"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "traces.jsonl"
RESULTS_DIR = ROOT / "results"

# ChromaDB collection
COLLECTION = "fastapi_docs"

# Ollama models already installed on your Mac
EMBED_MODEL = "qwen3-embedding:0.6b"
GEN_MODEL = "qwen2.5:7b"
JUDGE_MODEL = "llama3.1:8b"

# Hugging Face cross-encoder reranker
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval configuration
TOP_K_RETRIEVE = 20
TOP_K_FINAL = 5

# Agent retry configuration
MAX_RETRIES = 2
MAX_REGENERATIONS = 1

# Minimum faithfulness score, from 0 to 1
FAITHFULNESS_MIN = 0.8

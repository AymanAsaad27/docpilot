from functools import lru_cache

import chromadb
from sentence_transformers import CrossEncoder

from src.config import (
    CHROMA_DIR,
    COLLECTION,
    RERANK_MODEL,
    TOP_K_FINAL,
    TOP_K_RETRIEVE,
)
from src.llm import embed


@lru_cache(maxsize=1)
def get_collection():
    """Open the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load and cache the cross-encoder reranking model."""
    return CrossEncoder(RERANK_MODEL)


def retrieve_candidates(query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
    """Retrieve candidate passages using vector similarity."""
    query = query.strip()
    if not query:
        return []

    collection = get_collection()
    query_embedding = embed([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    candidates = []
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc_id, document, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        candidates.append({
            "id": doc_id,
            "text": document,
            "metadata": metadata,
            "vector_distance": float(distance),
            "vector_score": 1.0 - float(distance),
        })

    return candidates


def rerank_candidates(
    query: str,
    candidates: list[dict],
    top_k: int = TOP_K_FINAL,
) -> list[dict]:
    """Rerank retrieved passages with a cross-encoder."""
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(query, candidate["text"]) for candidate in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)

    ranked = []
    for candidate, score in zip(candidates, scores):
        item = candidate.copy()
        item["rerank_score"] = float(score)
        ranked.append(item)

    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return ranked[:top_k]


def search(
    query: str,
    retrieve_k: int = TOP_K_RETRIEVE,
    final_k: int = TOP_K_FINAL,
) -> list[dict]:
    """Retrieve and rerank passages for a user question."""
    candidates = retrieve_candidates(query, top_k=retrieve_k)
    return rerank_candidates(query, candidates, top_k=final_k)

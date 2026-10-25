import re
from pathlib import Path

import chromadb

from src.config import CHROMA_DIR, COLLECTION, DOCS_DIR
from src.llm import embed


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
BATCH_SIZE = 32

EXCLUDED_FILES = {"_llm-test.md"}


def read_markdown(path: Path) -> str:
    """Read a Markdown document as UTF-8 text."""
    return path.read_text(encoding="utf-8", errors="replace")


def split_by_headings(text: str) -> list[tuple[str, str]]:
    """Split Markdown by headings, ignoring headings inside fenced code."""
    sections = []
    current_heading = "Introduction"
    current_lines = []
    in_code_fence = False
    fence_char = None

    for line in text.splitlines():
        fence_match = re.match(r"^\s*(`{3,}|~{3,})", line)

        if fence_match:
            marker = fence_match.group(1)
            if not in_code_fence:
                in_code_fence = True
                fence_char = marker[0]
            elif marker[0] == fence_char:
                in_code_fence = False
                fence_char = None

        heading_match = (
            None
            if in_code_fence
            else re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        )

        if heading_match:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append((current_heading, content))
            current_heading = heading_match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    content = "\n".join(current_lines).strip()
    if content:
        sections.append((current_heading, content))

    return sections


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping character-based chunks."""
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))

        if end < len(text):
            boundary = max(
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
            )
            if boundary > start + size // 2:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(start + 1, end - overlap)

    return chunks


def build_chunks() -> list[dict]:
    """Load Markdown files and produce chunks with source metadata."""
    if not DOCS_DIR.exists():
        raise FileNotFoundError(
            f"Documentation directory not found: {DOCS_DIR}"
        )

    files = sorted(
        path
        for path in DOCS_DIR.rglob("*.md")
        if path.name not in EXCLUDED_FILES
    )

    if not files:
        raise FileNotFoundError(f"No Markdown files found in {DOCS_DIR}")

    documents = []

    for path in files:
        relative_path = path.relative_to(DOCS_DIR).as_posix()
        text = read_markdown(path)

        for section_index, (heading, section_text) in enumerate(
            split_by_headings(text)
        ):
            heading_prefix = f"Section: {heading}\n\n"
            available_size = CHUNK_SIZE - len(heading_prefix)

            if available_size <= 0:
                heading_prefix = f"Section: {heading[:100]}\n\n"
                available_size = CHUNK_SIZE - len(heading_prefix)

            section_chunks = chunk_text(
                section_text,
                size=available_size,
                overlap=min(CHUNK_OVERLAP, available_size - 1),
            )

            for chunk_index, chunk in enumerate(section_chunks):
                documents.append({
                    "id": (
                        f"{relative_path}::section-{section_index}"
                        f"::chunk-{chunk_index}"
                    ),
                    "text": heading_prefix + chunk,
                    "metadata": {
                        "source": relative_path,
                        "heading": heading,
                        "section_index": section_index,
                        "chunk_index": chunk_index,
                    },
                })

    return documents


def ingest(reset: bool = False) -> None:
    """Embed document chunks and store them in persistent ChromaDB."""
    documents = build_chunks()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception as exc:
            # A missing collection is fine; other errors should be visible.
            if "does not exist" not in str(exc).lower():
                raise

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"Markdown files indexed: {len(set(d['metadata']['source'] for d in documents))}")
    print(f"Chunks prepared: {len(documents)}")

    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start:start + BATCH_SIZE]
        vectors = embed([item["text"] for item in batch])

        collection.upsert(
            ids=[item["id"] for item in batch],
            documents=[item["text"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
            embeddings=vectors,
        )

        processed = min(start + len(batch), len(documents))
        print(f"Indexed {processed}/{len(documents)} chunks")

    print(f"\nIngestion complete. Collection count: {collection.count()}")
    print(f"ChromaDB path: {CHROMA_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index FastAPI documentation.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing collection before rebuilding it.",
    )
    args = parser.parse_args()
    ingest(reset=args.reset)

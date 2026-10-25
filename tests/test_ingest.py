from pathlib import Path

import pytest

from src import ingest


def test_read_markdown_reads_utf8(tmp_path):
    file_path = tmp_path / "example.md"
    file_path.write_text("# Café\n\nこんにちは", encoding="utf-8")

    result = ingest.read_markdown(file_path)

    assert result == "# Café\n\nこんにちは"


def test_split_by_headings_preserves_section_content():
    text = (
        "Introduction text.\n"
        "\n"
        "## First section\n"
        "First section content.\n"
        "\n"
        "### Second section\n"
        "Second section content."
    )

    sections = ingest.split_by_headings(text)

    assert sections == [
        ("Introduction", "Introduction text."),
        ("First section", "First section content."),
        ("Second section", "Second section content."),
    ]


def test_split_by_headings_ignores_headings_inside_backtick_code():
    text = (
        "## Real heading\n"
        "Some explanation.\n"
        "\n"
        "```python\n"
        "## This is code, not a heading\n"
        "print('hello')\n"
        "```\n"
        "\n"
        "## Another real heading\n"
        "More explanation."
    )

    sections = ingest.split_by_headings(text)

    assert [heading for heading, _ in sections] == [
        "Real heading",
        "Another real heading",
    ]
    assert "## This is code, not a heading" in sections[0][1]


def test_split_by_headings_ignores_headings_inside_tilde_code():
    text = (
        "## Example\n"
        "~~~python\n"
        "# Not a Markdown section\n"
        "~~~\n"
        "\n"
        "## Next section\n"
        "Content."
    )

    sections = ingest.split_by_headings(text)

    assert [heading for heading, _ in sections] == [
        "Example",
        "Next section",
    ]
    assert "# Not a Markdown section" in sections[0][1]


def test_chunk_text_returns_empty_for_blank_text():
    assert ingest.chunk_text("   \n  ") == []


def test_chunk_text_rejects_invalid_size_and_overlap():
    with pytest.raises(ValueError, match="size must be positive"):
        ingest.chunk_text("some text", size=0)

    with pytest.raises(
        ValueError,
        match="overlap must be non-negative and smaller than size",
    ):
        ingest.chunk_text("some text", size=10, overlap=10)

    with pytest.raises(
        ValueError,
        match="overlap must be non-negative and smaller than size",
    ):
        ingest.chunk_text("some text", size=10, overlap=-1)


def test_chunk_text_splits_long_text_with_overlap():
    text = "A" * 100 + "\n\n" + "B" * 100

    chunks = ingest.chunk_text(text, size=80, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 80 for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)


def test_build_chunks_excludes_test_document_and_adds_metadata(
    tmp_path, monkeypatch
):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "guide.md").write_text(
        "# Introduction\n\nIntro text.\n\n"
        "## Parameters\n\nDefault values are supported.",
        encoding="utf-8",
    )
    (docs_dir / "_llm-test.md").write_text(
        "# Internal test\n\nThis should not be indexed.",
        encoding="utf-8",
    )

    monkeypatch.setattr(ingest, "DOCS_DIR", docs_dir)

    documents = ingest.build_chunks()

    assert documents
    assert all(
        item["metadata"]["source"] == "guide.md"
        for item in documents
    )
    assert all(
        "_llm-test.md" not in item["id"]
        for item in documents
    )
    assert all(
        item["text"].startswith(
            f"Section: {item['metadata']['heading']}\n\n"
        )
        for item in documents
    )
    assert all(
        "section_index" in item["metadata"]
        and "chunk_index" in item["metadata"]
        for item in documents
    )


def test_build_chunks_raises_when_docs_directory_is_missing(
    tmp_path, monkeypatch
):
    missing_dir = tmp_path / "missing_docs"
    monkeypatch.setattr(ingest, "DOCS_DIR", missing_dir)

    with pytest.raises(FileNotFoundError, match="Documentation directory"):
        ingest.build_chunks()


def test_build_chunks_raises_when_no_markdown_files_exist(
    tmp_path, monkeypatch
):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "notes.txt").write_text("Not Markdown", encoding="utf-8")
    (docs_dir / "_llm-test.md").write_text(
        "# Excluded",
        encoding="utf-8",
    )

    monkeypatch.setattr(ingest, "DOCS_DIR", docs_dir)

    with pytest.raises(FileNotFoundError, match="No Markdown files found"):
        ingest.build_chunks()

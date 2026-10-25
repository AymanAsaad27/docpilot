from unittest.mock import Mock

import pytest

from src import ingest


def make_document(index):
    return {
        "id": f"doc-{index}",
        "text": f"Document text {index}",
        "metadata": {
            "source": f"guide-{index}.md",
            "heading": "Introduction",
            "section_index": 0,
            "chunk_index": 0,
        },
    }


@pytest.fixture
def ingestion_mocks(monkeypatch):
    documents = [make_document(i) for i in range(3)]

    collection = Mock()
    collection.count.return_value = 3

    client = Mock()
    client.get_or_create_collection.return_value = collection

    monkeypatch.setattr(ingest, "build_chunks", lambda: documents)
    monkeypatch.setattr(
        ingest.chromadb,
        "PersistentClient",
        lambda path: client,
    )

    embedding = Mock(
        side_effect=[
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        ]
    )
    monkeypatch.setattr(ingest, "embed", embedding)

    return {
        "documents": documents,
        "collection": collection,
        "client": client,
        "embed": embedding,
    }


def test_ingest_indexes_documents_and_prints_summary(
    ingestion_mocks, capsys
):
    mocks = ingestion_mocks

    ingest.ingest()

    mocks["client"].get_or_create_collection.assert_called_once_with(
        name=ingest.COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    mocks["embed"].assert_called_once_with(
        [item["text"] for item in mocks["documents"]]
    )

    mocks["collection"].upsert.assert_called_once_with(
        ids=[item["id"] for item in mocks["documents"]],
        documents=[item["text"] for item in mocks["documents"]],
        metadatas=[item["metadata"] for item in mocks["documents"]],
        embeddings=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
    )

    output = capsys.readouterr().out
    assert "Markdown files indexed: 3" in output
    assert "Chunks prepared: 3" in output
    assert "Indexed 3/3 chunks" in output
    assert "Ingestion complete. Collection count: 3" in output
    assert str(ingest.CHROMA_DIR) in output


def test_ingest_reset_deletes_existing_collection(
    ingestion_mocks
):
    mocks = ingestion_mocks

    ingest.ingest(reset=True)

    mocks["client"].delete_collection.assert_called_once_with(
        ingest.COLLECTION
    )
    mocks["client"].get_or_create_collection.assert_called_once()


def test_ingest_reset_ignores_missing_collection_error(
    ingestion_mocks
):
    mocks = ingestion_mocks
    mocks["client"].delete_collection.side_effect = Exception(
        "Collection does not exist"
    )

    ingest.ingest(reset=True)

    mocks["client"].get_or_create_collection.assert_called_once()


def test_ingest_reset_reraises_unexpected_delete_error(
    ingestion_mocks
):
    mocks = ingestion_mocks
    mocks["client"].delete_collection.side_effect = RuntimeError(
        "Database connection failed"
    )

    with pytest.raises(
        RuntimeError,
        match="Database connection failed",
    ):
        ingest.ingest(reset=True)

    mocks["client"].get_or_create_collection.assert_not_called()


def test_ingest_processes_multiple_batches(
    monkeypatch, capsys
):
    documents = [make_document(i) for i in range(ingest.BATCH_SIZE + 1)]

    collection = Mock()
    collection.count.return_value = len(documents)

    client = Mock()
    client.get_or_create_collection.return_value = collection

    monkeypatch.setattr(ingest, "build_chunks", lambda: documents)
    monkeypatch.setattr(
        ingest.chromadb,
        "PersistentClient",
        lambda path: client,
    )
    monkeypatch.setattr(
        ingest,
        "embed",
        lambda texts: [[0.1, 0.2] for _ in texts],
    )

    ingest.ingest()

    assert collection.upsert.call_count == 2

    first_batch = collection.upsert.call_args_list[0].kwargs
    second_batch = collection.upsert.call_args_list[1].kwargs

    assert len(first_batch["ids"]) == ingest.BATCH_SIZE
    assert len(second_batch["ids"]) == 1

    output = capsys.readouterr().out
    assert f"Indexed {ingest.BATCH_SIZE}/{len(documents)} chunks" in output
    assert f"Indexed {len(documents)}/{len(documents)} chunks" in output


def test_ingest_does_not_create_collection_when_build_fails(
    monkeypatch
):
    client_constructor = Mock()
    monkeypatch.setattr(
        ingest,
        "build_chunks",
        Mock(side_effect=FileNotFoundError("No documentation")),
    )
    monkeypatch.setattr(
        ingest.chromadb,
        "PersistentClient",
        client_constructor,
    )

    with pytest.raises(FileNotFoundError, match="No documentation"):
        ingest.ingest()

    client_constructor.assert_not_called()


def test_ingest_propagates_embedding_errors(
    ingestion_mocks
):
    mocks = ingestion_mocks
    mocks["embed"].side_effect = RuntimeError("Embedding service failed")

    with pytest.raises(
        RuntimeError,
        match="Embedding service failed",
    ):
        ingest.ingest()

    mocks["collection"].upsert.assert_not_called()

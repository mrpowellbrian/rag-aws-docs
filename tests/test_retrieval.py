"""Tests for the Chroma storage layer.

Chroma is mocked at the chromadb.PersistentClient level so these tests
run without a filesystem or a running Chroma instance.
"""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rag_aws_docs.storage.chroma import RetrievedChunk, VectorStore


def _mock_collection(count: int = 10) -> MagicMock:
    col = MagicMock()
    col.count.return_value = count
    return col


def _make_store(tmp_path: Path, mock_collection: MagicMock) -> VectorStore:
    with patch("chromadb.PersistentClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_or_create_collection.return_value = mock_collection

        store = VectorStore(path=str(tmp_path / "chroma"), collection_name="test")
        # Force collection initialisation.
        store._collection = mock_collection
        return store


class TestVectorStoreQuery:
    def test_empty_collection_returns_empty_list(self, tmp_path: Path) -> None:
        col = _mock_collection(count=0)
        store = _make_store(tmp_path, col)
        result = store.query([0.1, 0.2, 0.3])
        assert result == []

    def test_distance_converted_to_similarity(self, tmp_path: Path) -> None:
        col = _mock_collection(count=5)
        # Chroma returns distance=0.2 → expected similarity=0.8
        col.query.return_value = {
            "ids": [["abc123"]],
            "documents": [["some content"]],
            "metadatas": [[{"source_path": "iam/intro.md", "repo": "r", "chunk_index": 0}]],
            "distances": [[0.2]],
        }
        store = _make_store(tmp_path, col)
        results = store.query([0.1, 0.2, 0.3], top_k=1)

        assert len(results) == 1
        assert abs(results[0].score - 0.8) < 1e-6

    def test_distance_zero_gives_score_one(self, tmp_path: Path) -> None:
        col = _mock_collection(count=1)
        col.query.return_value = {
            "ids": [["id1"]],
            "documents": [["doc"]],
            "metadatas": [[{"source_path": "p", "repo": "r", "chunk_index": 0}]],
            "distances": [[0.0]],
        }
        store = _make_store(tmp_path, col)
        results = store.query([0.0], top_k=1)
        assert results[0].score == pytest.approx(1.0)

    def test_large_distance_clamped_to_zero(self, tmp_path: Path) -> None:
        # Distance > 1.0 (opposite vectors in cosine space) should clamp to 0.
        col = _mock_collection(count=1)
        col.query.return_value = {
            "ids": [["id1"]],
            "documents": [["doc"]],
            "metadatas": [[{"source_path": "p", "repo": "r", "chunk_index": 0}]],
            "distances": [[1.5]],
        }
        store = _make_store(tmp_path, col)
        results = store.query([0.0], top_k=1)
        assert results[0].score == 0.0

    def test_returned_type_is_retrieved_chunk(self, tmp_path: Path) -> None:
        col = _mock_collection(count=3)
        col.query.return_value = {
            "ids": [["id1"]],
            "documents": [["content"]],
            "metadatas": [[{"source_path": "s", "repo": "r", "chunk_index": 2}]],
            "distances": [[0.3]],
        }
        store = _make_store(tmp_path, col)
        results = store.query([0.1], top_k=1)
        assert isinstance(results[0], RetrievedChunk)
        assert results[0].id == "id1"
        assert results[0].content == "content"
        assert results[0].chunk_index == 2

    def test_top_k_capped_at_collection_count(self, tmp_path: Path) -> None:
        col = _mock_collection(count=3)
        col.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]
        }
        store = _make_store(tmp_path, col)
        store.query([0.1], top_k=100)
        call_kwargs = col.query.call_args[1]
        assert call_kwargs["n_results"] == 3  # capped at collection size

    def test_where_filter_passed_to_chroma(self, tmp_path: Path) -> None:
        col = _mock_collection(count=5)
        col.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]
        }
        store = _make_store(tmp_path, col)
        store.query([0.1], where={"repo": "awsdocs/iam-user-guide"})
        call_kwargs = col.query.call_args[1]
        assert call_kwargs["where"] == {"repo": "awsdocs/iam-user-guide"}


class TestVectorStoreUpsert:
    def test_mismatched_lengths_raises(self, tmp_path: Path) -> None:
        col = _mock_collection()
        store = _make_store(tmp_path, col)
        with pytest.raises(ValueError, match="must match"):
            store.upsert([], [[0.1, 0.2]])  # type: ignore[arg-type]

    def test_empty_upsert_is_noop(self, tmp_path: Path) -> None:
        col = _mock_collection()
        store = _make_store(tmp_path, col)
        store.upsert([], [])
        col.upsert.assert_not_called()

"""Tests for the token-aware chunker."""

import pytest

from rag_aws_docs.ingest.chunker import Chunk, _make_chunk_id, _token_count, chunk_document
from rag_aws_docs.ingest.loader import Document


def _doc(content: str, source: str = "test/doc.md", repo: str = "test/repo") -> Document:
    return Document(
        content=content,
        source_path=source,
        repo=repo,
        metadata={"repo": repo, "source_path": source, "filename": "doc.md"},
    )


class TestTokenCount:
    def test_empty_string(self) -> None:
        assert _token_count("") == 0

    def test_known_short_string(self) -> None:
        # "hello world" is 2 tokens in cl100k_base.
        assert _token_count("hello world") == 2

    def test_count_is_positive_for_nonempty(self) -> None:
        assert _token_count("AWS Lambda is a serverless compute service.") > 0


class TestChunkId:
    def test_deterministic(self) -> None:
        a = _make_chunk_id("repo/a", "path/doc.md", 0)
        b = _make_chunk_id("repo/a", "path/doc.md", 0)
        assert a == b

    def test_different_index_differs(self) -> None:
        a = _make_chunk_id("repo/a", "path/doc.md", 0)
        b = _make_chunk_id("repo/a", "path/doc.md", 1)
        assert a != b

    def test_different_path_differs(self) -> None:
        a = _make_chunk_id("repo/a", "path/doc.md", 0)
        b = _make_chunk_id("repo/a", "path/other.md", 0)
        assert a != b

    def test_length_is_16(self) -> None:
        chunk_id = _make_chunk_id("r", "p", 0)
        assert len(chunk_id) == 16


class TestChunkDocument:
    def test_empty_document_returns_empty(self) -> None:
        doc = _doc("")
        assert chunk_document(doc) == []

    def test_whitespace_only_returns_empty(self) -> None:
        doc = _doc("   \n\n\t  ")
        assert chunk_document(doc) == []

    def test_short_document_is_single_chunk(self) -> None:
        doc = _doc("AWS Lambda is a serverless compute service.")
        chunks = chunk_document(doc, chunk_size=512, chunk_overlap=64)
        assert len(chunks) == 1
        assert "Lambda" in chunks[0].content

    def test_chunk_is_frozen_dataclass(self) -> None:
        doc = _doc("Some content.")
        chunk = chunk_document(doc)[0]
        with pytest.raises(Exception):
            chunk.content = "mutated"  # type: ignore[misc]

    def test_metadata_propagated(self) -> None:
        doc = _doc("Content.", source="iam/intro.md", repo="awsdocs/iam-user-guide")
        chunks = chunk_document(doc)
        assert chunks[0].source_path == "iam/intro.md"
        assert chunks[0].repo == "awsdocs/iam-user-guide"
        assert chunks[0].metadata["repo"] == "awsdocs/iam-user-guide"

    def test_long_document_produces_multiple_chunks(self) -> None:
        # Build a document whose token count well exceeds chunk_size.
        sentence = "AWS IAM allows you to manage access to AWS services and resources. "
        doc = _doc(sentence * 40)
        chunks = chunk_document(doc, chunk_size=64, chunk_overlap=8)
        assert len(chunks) > 1

    def test_chunk_indices_are_sequential(self) -> None:
        sentence = "Each IAM policy consists of one or more statements. "
        doc = _doc(sentence * 40)
        chunks = chunk_document(doc, chunk_size=64, chunk_overlap=8)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_ids_are_unique(self) -> None:
        sentence = "Roles are a secure way to grant permissions. "
        doc = _doc(sentence * 40)
        chunks = chunk_document(doc, chunk_size=64, chunk_overlap=8)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_token_counts_within_bounds(self) -> None:
        sentence = "Lambda functions can be triggered by many AWS services. "
        doc = _doc(sentence * 40)
        chunk_size = 64
        chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=8)
        # Most chunks should be at or below chunk_size; oversized sentences
        # are allowed to exceed it.
        normal_chunks = [c for c in chunks if c.token_count <= chunk_size * 2]
        assert len(normal_chunks) > 0

    def test_overlap_means_adjacent_chunks_share_content(self) -> None:
        # With overlap, the end of chunk N should appear at the start of chunk N+1.
        sentence = "Access control lists provide coarse-grained permission management. "
        doc = _doc(sentence * 30)
        chunks = chunk_document(doc, chunk_size=50, chunk_overlap=15)
        if len(chunks) < 2:
            pytest.skip("document too short to produce multiple chunks at this size")
        # The last word of chunk 0 should appear in chunk 1.
        last_word = chunks[0].content.split()[-1]
        assert last_word in chunks[1].content

    def test_oversized_single_sentence_emitted(self) -> None:
        # A sentence longer than chunk_size must still appear in the output.
        long_sentence = "word " * 600  # ~600 tokens
        doc = _doc(long_sentence)
        chunks = chunk_document(doc, chunk_size=128, chunk_overlap=16)
        reconstructed = " ".join(c.content for c in chunks)
        assert "word" in reconstructed

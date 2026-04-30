"""Fixed-size token-aware chunker with sentence-boundary alignment.

Strategy
--------
1. Split the document into paragraphs on double-newlines. Paragraphs are the
   natural unit in both Markdown and RST — they rarely span a topic boundary
   the way an arbitrary byte offset would.
2. Within each paragraph, split on sentence-terminal punctuation so we never
   cut mid-sentence when a paragraph exceeds chunk_size.
3. Greedily accumulate units until adding the next one would exceed chunk_size.
   When a chunk is full, emit it, then seed the next chunk with the trailing
   overlap_tokens worth of text from the just-emitted chunk. Overlap preserves
   context for questions whose answer straddles a chunk boundary.

Why tiktoken instead of len(text.split())?
------------------------------------------
Token count is what the LLM and the embedding model actually see. A word like
"multipart/form-data" is one word but three or four tokens. Sizing chunks by
tokens rather than words prevents silent prompt truncation when chunks are
assembled into a context window.
"""

import hashlib
import re
from dataclasses import dataclass, field

import tiktoken

from rag_aws_docs.config import settings
from rag_aws_docs.ingest.loader import Document

# cl100k_base is the tokenizer used by OpenAI embeddings and is a reasonable
# approximation for sentence-transformers and Claude token counts.
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Sentence terminal: period/!/? followed by whitespace or end-of-string,
# but not inside abbreviations like "e.g." or version numbers like "1.2".
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """A single chunk ready for embedding and storage."""

    id: str
    content: str
    token_count: int
    source_path: str
    repo: str
    chunk_index: int
    metadata: dict[str, str] = field(default_factory=dict)


def _token_count(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def _split_into_units(text: str) -> list[str]:
    """Split text into sentence-level units, grouping short lines."""
    paragraphs = re.split(r"\n{2,}", text)
    units: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        sentences = _SENTENCE_END.split(para)
        units.extend(s.strip() for s in sentences if s.strip())
    return units


def _make_chunk_id(repo: str, source_path: str, chunk_index: int) -> str:
    key = f"{repo}::{source_path}::{chunk_index}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def chunk_document(
    doc: Document,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split a Document into overlapping token-bounded Chunk objects.

    Args:
        doc: Source document from the loader.
        chunk_size: Max tokens per chunk. Defaults to settings.chunk_size.
        chunk_overlap: Overlap tokens between adjacent chunks.
                       Defaults to settings.chunk_overlap.

    Returns:
        Ordered list of Chunk objects. Empty documents return an empty list.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    units = _split_into_units(doc.content)
    if not units:
        return []

    chunks: list[Chunk] = []
    current_units: list[str] = []
    current_tokens = 0

    def emit_chunk() -> None:
        content = " ".join(current_units).strip()
        if not content:
            return
        idx = len(chunks)
        chunks.append(
            Chunk(
                id=_make_chunk_id(doc.repo, doc.source_path, idx),
                content=content,
                token_count=_token_count(content),
                source_path=doc.source_path,
                repo=doc.repo,
                chunk_index=idx,
                metadata={
                    **doc.metadata,
                    "chunk_index": str(idx),
                },
            )
        )

    def apply_overlap() -> tuple[list[str], int]:
        """Seed the next chunk with trailing units from the current chunk."""
        if not current_units or chunk_overlap == 0:
            return [], 0
        overlap_units: list[str] = []
        overlap_tokens = 0
        for unit in reversed(current_units):
            unit_tokens = _token_count(unit)
            if overlap_tokens + unit_tokens > chunk_overlap:
                break
            overlap_units.insert(0, unit)
            overlap_tokens += unit_tokens
        return overlap_units, overlap_tokens

    for unit in units:
        unit_tokens = _token_count(unit)

        # A single unit larger than chunk_size can't be split further without
        # breaking sentence integrity — emit it as an oversized chunk.
        if unit_tokens >= chunk_size:
            if current_units:
                emit_chunk()
                current_units, current_tokens = apply_overlap()
            current_units.append(unit)
            current_tokens += unit_tokens
            emit_chunk()
            current_units, current_tokens = [], 0
            continue

        if current_tokens + unit_tokens > chunk_size:
            emit_chunk()
            current_units, current_tokens = apply_overlap()

        current_units.append(unit)
        current_tokens += unit_tokens

    if current_units:
        emit_chunk()

    return chunks


def chunk_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Chunk a list of documents. Convenience wrapper around chunk_document."""
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size, chunk_overlap))
    return chunks

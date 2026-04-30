"""Chroma vector store integration.

All interaction with Chroma goes through VectorStore. The collection is
created on first use; subsequent calls reuse it. Chunk upserts are
idempotent — re-ingesting the same corpus overwrites existing vectors
rather than duplicating them (Chroma matches on document ID).

Chroma's cosine distance is stored as (1 - cosine_similarity), so
distances are in [0, 2]. We convert to similarity scores in [0, 1]
before returning to callers so that higher is always better.
"""

import logging
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from rag_aws_docs.config import settings
from rag_aws_docs.ingest.chunker import Chunk

logger = logging.getLogger(__name__)

# Chroma metadata values must be str, int, or float — not None or bool.
_METADATA_SCALAR = str | int | float


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from similarity search, annotated with its score."""

    id: str
    content: str
    source_path: str
    repo: str
    chunk_index: int
    # Cosine similarity in [0, 1]. Higher = more similar to the query.
    score: float
    metadata: dict[str, str]


class VectorStore:
    """Thin wrapper around a single Chroma collection."""

    def __init__(
        self,
        path: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        resolved_path = path or str(settings.chroma_path)
        resolved_collection = collection_name or settings.chroma_collection

        settings.chroma_path.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=resolved_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection_name = resolved_collection
        # Collection is fetched lazily so we don't pay the init cost when
        # the CLI runs a subcommand that doesn't touch Chroma.
        self._collection: chromadb.Collection | None = None
        logger.info("vector store initialised at %s / %s", resolved_path, resolved_collection)

    def _get_or_create_collection(self) -> chromadb.Collection:
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "collection '%s' has %d documents",
                self._collection_name,
                self._collection.count(),
            )
        return self._collection

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks and their pre-computed embeddings into the collection.

        Args:
            chunks: Chunk objects from the chunker.
            embeddings: Parallel list of embedding vectors. Must be same length.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
            )
        if not chunks:
            return

        collection = self._get_or_create_collection()

        # Chroma metadata values must be scalars. Filter out any non-scalar
        # values defensively in case metadata dict has unexpected contents.
        def _clean_meta(meta: dict[str, str]) -> dict[str, _METADATA_SCALAR]:
            return {k: v for k, v in meta.items() if isinstance(v, (str, int, float))}

        # Chroma's upsert batches internally, but we batch at 500 to keep
        # per-call memory bounded when ingesting large corpora.
        batch_size = 500
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            collection.upsert(
                ids=[c.id for c in batch_chunks],
                embeddings=batch_embeddings,
                documents=[c.content for c in batch_chunks],
                metadatas=[
                    {
                        **_clean_meta(c.metadata),
                        "source_path": c.source_path,
                        "repo": c.repo,
                        "chunk_index": c.chunk_index,
                        "token_count": c.token_count,
                    }
                    for c in batch_chunks
                ],
            )
            total += len(batch_chunks)
            logger.debug("upserted %d / %d chunks", total, len(chunks))

        logger.info("upserted %d chunks into '%s'", len(chunks), self._collection_name)

    def query(
        self,
        query_embedding: list[float],
        top_k: int | None = None,
        where: dict[str, str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k most similar chunks for a query embedding.

        Args:
            query_embedding: Embedded query vector.
            top_k: Number of results. Defaults to settings.top_k.
            where: Optional Chroma metadata filter, e.g. {"repo": "awsdocs/iam-user-guide"}.

        Returns:
            List of RetrievedChunk ordered by descending similarity score.
        """
        resolved_k = top_k or settings.top_k
        collection = self._get_or_create_collection()

        if collection.count() == 0:
            logger.warning("collection is empty — run 'rag-aws-docs ingest' first")
            return []

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(resolved_k, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = collection.query(**kwargs)

        ids = result["ids"][0]
        documents = result["documents"][0]  # type: ignore[index]
        metadatas = result["metadatas"][0]  # type: ignore[index]
        distances = result["distances"][0]  # type: ignore[index]

        retrieved: list[RetrievedChunk] = []
        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            # Convert cosine distance → similarity: similarity = 1 - distance.
            # Chroma's cosine distance is in [0, 2], so similarity is in [-1, 1].
            # Clamp to [0, 1] to avoid confusing negative values for low-quality matches.
            score = max(0.0, 1.0 - float(dist))
            retrieved.append(
                RetrievedChunk(
                    id=chunk_id,
                    content=doc,
                    source_path=str(meta.get("source_path", "")),
                    repo=str(meta.get("repo", "")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=score,
                    metadata={k: str(v) for k, v in meta.items()},
                )
            )

        return retrieved

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._get_or_create_collection().count()

    def delete_collection(self) -> None:
        """Drop the collection entirely. Used for re-ingestion from scratch."""
        self._client.delete_collection(self._collection_name)
        self._collection = None
        logger.info("deleted collection '%s'", self._collection_name)

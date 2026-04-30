"""Embedding provider abstraction.

The embedding model is the one component that cannot be swapped at query
time without re-ingesting the entire corpus — vectors from different models
live in incompatible spaces. The abstraction exists so the corpus can be
re-ingested with a different provider (e.g., switching from local
sentence-transformers to OpenAI text-embedding-3-small for better recall)
without changing any other code.

Providers
---------
local   sentence-transformers model, runs on CPU, no API cost.
        Default model: all-MiniLM-L6-v2 (384-dim, ~22M params, fast).
openai  Requires OPENAI_API_KEY and the [openai] extra.
        Default model: text-embedding-3-small (1536-dim).
"""

import logging
from abc import ABC, abstractmethod

from rag_aws_docs.config import EmbeddingProvider, settings

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the output vectors."""
        ...


class LocalEmbedder(BaseEmbedder):
    """sentence-transformers embedder. Model is loaded once and cached."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name or settings.embedding_model
        logger.info("loading local embedding model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info("embedding dimension: %d", self._dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [v.tolist() for v in vectors]

    def dimension(self) -> int:
        return int(self._dim)


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings. Requires OPENAI_API_KEY and pip install rag-aws-docs[openai]."""

    # OpenAI's API accepts up to 2048 inputs per request, but batching at 512
    # keeps individual request latency reasonable and makes retries cheaper.
    BATCH_SIZE = 512

    def __init__(self, model_name: str | None = None) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "OpenAI embeddings require the [openai] extra: "
                "pip install rag-aws-docs[openai]"
            ) from e

        self._model_name = model_name or settings.embedding_model
        self._client = openai.OpenAI()
        # Dimension varies by model; resolve lazily on first embed call.
        self._dim: int | None = None
        logger.info("using OpenAI embedding model: %s", self._model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        import openai

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            try:
                response = self._client.embeddings.create(
                    input=batch,
                    model=self._model_name,
                )
            except openai.OpenAIError as exc:
                logger.error("OpenAI embedding request failed: %s", exc)
                raise

            vectors = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
            all_vectors.extend(vectors)

            if self._dim is None and vectors:
                self._dim = len(vectors[0])

        return all_vectors

    def dimension(self) -> int:
        if self._dim is None:
            # Resolve by embedding a single probe string.
            probe = self.embed(["probe"])
            self._dim = len(probe[0])
        return self._dim


# ── Factory ───────────────────────────────────────────────────────────────────

_instance: BaseEmbedder | None = None


def get_embedder(provider: EmbeddingProvider | None = None) -> BaseEmbedder:
    """Return the configured embedder, constructing it once per process.

    The singleton is intentional: SentenceTransformer model loading takes
    several seconds and allocates significant memory. Constructing it on
    every call would make the CLI unusably slow.
    """
    global _instance
    if _instance is not None:
        return _instance

    resolved = provider or settings.embedding_provider

    if resolved == EmbeddingProvider.LOCAL:
        _instance = LocalEmbedder()
    elif resolved == EmbeddingProvider.OPENAI:
        _instance = OpenAIEmbedder()
    else:
        raise ValueError(f"unknown embedding provider: {resolved}")

    return _instance

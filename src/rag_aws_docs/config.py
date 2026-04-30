from enum import StrEnum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProvider(StrEnum):
    LOCAL = "local"
    OPENAI = "openai"


class LLMProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"


# Corpus entries pulled from public awsdocs GitHub repositories.
# Each tuple is (repo_slug, subfolder_within_repo).
DEFAULT_CORPUS: list[tuple[str, str]] = [
    ("awsdocs/iam-user-guide", "doc_source"),
    ("awsdocs/aws-lambda-developer-guide", "doc_source"),
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", repr=False)
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    llm_model: str = "claude-haiku-4-5-20251001"

    # Pricing in USD per million tokens — used for cost logging.
    # Update when Anthropic publishes new pricing.
    llm_input_cost_per_mtok: float = 0.80
    llm_output_cost_per_mtok: float = 4.00

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_provider: EmbeddingProvider = EmbeddingProvider.LOCAL
    embedding_model: str = "all-MiniLM-L6-v2"
    # OpenAI embedding pricing (only relevant when embedding_provider=openai)
    embedding_cost_per_mtok: float = 0.02

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512, ge=64, le=2048)
    chunk_overlap: int = Field(default=64, ge=0, le=512)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = Field(default=5, ge=1, le=20)

    # ── Storage ───────────────────────────────────────────────────────────────
    chroma_path: Path = Path("./chroma_db")
    chroma_collection: str = "aws_docs"

    # ── Ingestion ─────────────────────────────────────────────────────────────
    # Local directory where corpus repos are cloned.
    data_path: Path = Path("./data/repos")

    # ── Observability ─────────────────────────────────────────────────────────
    log_file: Path = Path("./rag_queries.jsonl")

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_lt_chunk_size(cls, v: int, info: object) -> int:
        # info.data may not contain chunk_size if it failed validation, so guard.
        data = getattr(info, "data", {})
        chunk_size = data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})")
        return v

    @field_validator("anthropic_api_key")
    @classmethod
    def mask_key_in_repr(cls, v: str) -> str:
        # Accepted empty so the CLI can print a useful error rather than crashing
        # at import time when no key is configured.
        return v


# Module-level singleton — import this everywhere rather than constructing
# Settings() multiple times (avoids re-reading .env on every import).
settings = Settings()

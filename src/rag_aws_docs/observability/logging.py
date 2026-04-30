"""Structured JSON query logging.

Every query writes one JSON line to settings.log_file. The log is the
source of truth for the metrics module — it can be replayed, grepped,
and fed into any log aggregator (CloudWatch, Datadog, etc.) without
schema changes.

Log record fields
-----------------
timestamp           ISO-8601 UTC
query               The raw query string
llm_model           Model ID used for generation
embedding_model     Embedding model used for retrieval
retrieved_chunks    List of {id, source_path, score} for each chunk
retrieval_latency_s Time from query embed to Chroma result, in seconds
generation_latency_s Time from context assembly to API response, in seconds
total_latency_s     End-to-end wall time
input_tokens        Exact count from API response
output_tokens       Exact count from API response
cost_usd            Calculated from token counts and pricing constants
min_score           Lowest retrieval score among returned chunks
max_score           Highest retrieval score among returned chunks
mean_score          Mean retrieval score — a proxy for retrieval confidence
"""

import json
import logging
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Generator

from rag_aws_docs.config import settings
from rag_aws_docs.generation.provider import GenerationResult
from rag_aws_docs.storage.chroma import RetrievedChunk

logger = logging.getLogger(__name__)


@contextmanager
def timer() -> Generator[dict[str, float], None, None]:
    """Context manager that records elapsed seconds into a shared dict."""
    record: dict[str, float] = {}
    start = time.perf_counter()
    yield record
    record["elapsed"] = time.perf_counter() - start


def _score_stats(chunks: list[RetrievedChunk]) -> dict[str, float]:
    if not chunks:
        return {"min_score": 0.0, "max_score": 0.0, "mean_score": 0.0}
    scores = [c.score for c in chunks]
    return {
        "min_score": round(min(scores), 4),
        "max_score": round(max(scores), 4),
        "mean_score": round(sum(scores) / len(scores), 4),
    }


def log_query(
    query: str,
    chunks: list[RetrievedChunk],
    result: GenerationResult,
    retrieval_latency: float,
    generation_latency: float,
    log_file: Path | None = None,
) -> None:
    """Append a structured JSON record for one query to the log file.

    Args:
        query: The raw user query string.
        chunks: Retrieved chunks, in the order returned by Chroma.
        result: GenerationResult from the LLM provider.
        retrieval_latency: Seconds spent on embedding + vector search.
        generation_latency: Seconds spent on LLM call.
        log_file: Override log path. Defaults to settings.log_file.
    """
    resolved_log = log_file or settings.log_file
    resolved_log.parent.mkdir(parents=True, exist_ok=True)

    score_stats = _score_stats(chunks)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "query": query,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "retrieved_chunks": [
            {"id": c.id, "source_path": c.source_path, "score": round(c.score, 4)}
            for c in chunks
        ],
        "retrieval_latency_s": round(retrieval_latency, 4),
        "generation_latency_s": round(generation_latency, 4),
        "total_latency_s": round(retrieval_latency + generation_latency, 4),
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": result.cost_usd,
        **score_stats,
    }

    with resolved_log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")

    # Surface retrieval quality issues immediately in stderr logs.
    if score_stats["max_score"] < 0.5:
        logger.warning(
            "low retrieval scores for query %r (max=%.3f) — "
            "consider re-ingesting or adjusting chunk_size",
            query[:80],
            score_stats["max_score"],
        )

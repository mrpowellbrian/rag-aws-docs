"""Cost and quality metrics derived from the query log.

Reads settings.log_file (a JSONL file) and computes summary statistics.
Designed to be called from the CLI (`rag-aws-docs metrics`) and from
tests with an injected log path.

All monetary values are in USD. Latency values are in seconds.
"""

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from rag_aws_docs.config import settings


@dataclass
class MetricsSummary:
    total_queries: int = 0
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    retrieval_latencies: list[float] = field(default_factory=list)
    generation_latencies: list[float] = field(default_factory=list)
    mean_scores: list[float] = field(default_factory=list)
    # Queries where max retrieval score < 0.5 — likely poor retrieval.
    low_quality_retrievals: int = 0

    @property
    def avg_cost_per_query(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.total_cost_usd / self.total_queries

    @property
    def p50_retrieval_latency(self) -> float:
        if not self.retrieval_latencies:
            return 0.0
        return statistics.median(self.retrieval_latencies)

    @property
    def p95_retrieval_latency(self) -> float:
        if not self.retrieval_latencies:
            return 0.0
        sorted_latencies = sorted(self.retrieval_latencies)
        idx = max(0, int(len(sorted_latencies) * 0.95) - 1)
        return sorted_latencies[idx]

    @property
    def p50_generation_latency(self) -> float:
        if not self.generation_latencies:
            return 0.0
        return statistics.median(self.generation_latencies)

    @property
    def p95_generation_latency(self) -> float:
        if not self.generation_latencies:
            return 0.0
        sorted_latencies = sorted(self.generation_latencies)
        idx = max(0, int(len(sorted_latencies) * 0.95) - 1)
        return sorted_latencies[idx]

    @property
    def avg_mean_score(self) -> float:
        if not self.mean_scores:
            return 0.0
        return statistics.mean(self.mean_scores)


def compute_metrics(log_file: Path | None = None) -> MetricsSummary:
    """Parse the query log and return aggregated metrics.

    Args:
        log_file: Path to the JSONL log. Defaults to settings.log_file.

    Returns:
        MetricsSummary. Returns an empty summary if the log does not exist.
    """
    resolved = log_file or settings.log_file
    summary = MetricsSummary()

    if not resolved.exists():
        return summary

    with resolved.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            summary.total_queries += 1
            summary.total_cost_usd += record.get("cost_usd", 0.0)
            summary.total_input_tokens += record.get("input_tokens", 0)
            summary.total_output_tokens += record.get("output_tokens", 0)

            if (r_lat := record.get("retrieval_latency_s")) is not None:
                summary.retrieval_latencies.append(r_lat)
            if (g_lat := record.get("generation_latency_s")) is not None:
                summary.generation_latencies.append(g_lat)
            if (ms := record.get("mean_score")) is not None:
                summary.mean_scores.append(ms)
            if record.get("max_score", 1.0) < 0.5:
                summary.low_quality_retrievals += 1

    return summary


def format_summary(summary: MetricsSummary) -> str:
    """Format a MetricsSummary as a human-readable table."""
    if summary.total_queries == 0:
        return "No queries logged yet. Run 'rag-aws-docs query' to get started."

    low_quality_pct = (
        summary.low_quality_retrievals / summary.total_queries * 100
        if summary.total_queries
        else 0.0
    )

    lines = [
        "Query log summary",
        "─" * 42,
        f"  Total queries              {summary.total_queries}",
        f"  Total cost                 ${summary.total_cost_usd:.4f}",
        f"  Avg cost per query         ${summary.avg_cost_per_query:.4f}",
        f"  Total input tokens         {summary.total_input_tokens:,}",
        f"  Total output tokens        {summary.total_output_tokens:,}",
        "",
        "Latency (seconds)",
        f"  Retrieval  p50={summary.p50_retrieval_latency:.3f}  "
        f"p95={summary.p95_retrieval_latency:.3f}",
        f"  Generation p50={summary.p50_generation_latency:.3f}  "
        f"p95={summary.p95_generation_latency:.3f}",
        "",
        "Retrieval quality",
        f"  Avg mean score             {summary.avg_mean_score:.3f}",
        f"  Low-quality retrievals     {summary.low_quality_retrievals} "
        f"({low_quality_pct:.1f}%)",
    ]
    return "\n".join(lines)

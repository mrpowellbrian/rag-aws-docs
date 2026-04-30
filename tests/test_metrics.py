"""Tests for the metrics aggregation module."""

import json
from pathlib import Path

import pytest

from rag_aws_docs.observability.metrics import MetricsSummary, compute_metrics, format_summary


def _write_log(tmp_path: Path, records: list[dict]) -> Path:
    log_file = tmp_path / "queries.jsonl"
    with log_file.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return log_file


def _record(**kwargs) -> dict:
    """Minimal valid log record with overridable fields."""
    base = {
        "timestamp": "2026-01-01T00:00:00+00:00",
        "query": "What is IAM?",
        "llm_model": "claude-haiku-4-5-20251001",
        "embedding_model": "all-MiniLM-L6-v2",
        "retrieved_chunks": [],
        "retrieval_latency_s": 0.1,
        "generation_latency_s": 0.5,
        "total_latency_s": 0.6,
        "input_tokens": 1000,
        "output_tokens": 200,
        "cost_usd": 0.0016,
        "min_score": 0.7,
        "max_score": 0.9,
        "mean_score": 0.8,
    }
    return {**base, **kwargs}


class TestComputeMetrics:
    def test_missing_log_returns_empty_summary(self, tmp_path: Path) -> None:
        summary = compute_metrics(tmp_path / "nonexistent.jsonl")
        assert summary.total_queries == 0
        assert summary.total_cost_usd == 0.0

    def test_empty_log_returns_empty_summary(self, tmp_path: Path) -> None:
        log_file = tmp_path / "queries.jsonl"
        log_file.write_text("")
        summary = compute_metrics(log_file)
        assert summary.total_queries == 0

    def test_single_record(self, tmp_path: Path) -> None:
        log_file = _write_log(tmp_path, [_record()])
        summary = compute_metrics(log_file)
        assert summary.total_queries == 1
        assert abs(summary.total_cost_usd - 0.0016) < 1e-9
        assert summary.total_input_tokens == 1000
        assert summary.total_output_tokens == 200

    def test_multiple_records_accumulate(self, tmp_path: Path) -> None:
        records = [_record(cost_usd=0.001, input_tokens=500)] * 4
        log_file = _write_log(tmp_path, records)
        summary = compute_metrics(log_file)
        assert summary.total_queries == 4
        assert abs(summary.total_cost_usd - 0.004) < 1e-9
        assert summary.total_input_tokens == 2000

    def test_avg_cost_per_query(self, tmp_path: Path) -> None:
        records = [
            _record(cost_usd=0.002),
            _record(cost_usd=0.004),
        ]
        log_file = _write_log(tmp_path, records)
        summary = compute_metrics(log_file)
        assert abs(summary.avg_cost_per_query - 0.003) < 1e-9

    def test_malformed_lines_are_skipped(self, tmp_path: Path) -> None:
        log_file = tmp_path / "queries.jsonl"
        log_file.write_text('{"valid": true, "cost_usd": 0.001}\nnot json\n{"cost_usd": 0.002}\n')
        # Should not raise, should skip the bad line.
        summary = compute_metrics(log_file)
        assert summary.total_queries == 2

    def test_low_quality_retrieval_counted(self, tmp_path: Path) -> None:
        records = [
            _record(max_score=0.9),  # good
            _record(max_score=0.3),  # low quality
            _record(max_score=0.4),  # low quality
        ]
        log_file = _write_log(tmp_path, records)
        summary = compute_metrics(log_file)
        assert summary.low_quality_retrievals == 2

    def test_latency_percentiles(self, tmp_path: Path) -> None:
        latencies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        records = [_record(retrieval_latency_s=lat) for lat in latencies]
        log_file = _write_log(tmp_path, records)
        summary = compute_metrics(log_file)
        assert summary.p50_retrieval_latency == pytest.approx(0.55, abs=0.1)
        assert summary.p95_retrieval_latency >= 0.9

    def test_mean_score_aggregation(self, tmp_path: Path) -> None:
        records = [
            _record(mean_score=0.6),
            _record(mean_score=0.8),
        ]
        log_file = _write_log(tmp_path, records)
        summary = compute_metrics(log_file)
        assert abs(summary.avg_mean_score - 0.7) < 1e-9


class TestFormatSummary:
    def test_empty_summary_message(self) -> None:
        summary = MetricsSummary()
        output = format_summary(summary)
        assert "No queries logged" in output

    def test_nonempty_summary_contains_key_fields(self, tmp_path: Path) -> None:
        records = [_record(cost_usd=0.003, input_tokens=2000, output_tokens=300)]
        log_file = _write_log(tmp_path, records)
        summary = compute_metrics(log_file)
        output = format_summary(summary)
        assert "Total queries" in output
        assert "Total cost" in output
        assert "Retrieval quality" in output

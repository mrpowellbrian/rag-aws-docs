"""Tests for cost calculation and GenerationResult."""

import pytest

from rag_aws_docs.generation.provider import GenerationResult, _calculate_cost


class TestCalculateCost:
    def test_zero_tokens_is_zero_cost(self) -> None:
        assert _calculate_cost(0, 0) == 0.0

    def test_known_values(self) -> None:
        # At default pricing: $0.80/MTok input, $4.00/MTok output
        # 1_000_000 input tokens = $0.80, 1_000_000 output tokens = $4.00
        cost = _calculate_cost(1_000_000, 1_000_000)
        assert abs(cost - 4.80) < 0.0001

    def test_typical_query_cost(self) -> None:
        # Typical query: ~2780 input tokens, ~300 output tokens
        # Input: (2780 / 1_000_000) * 0.80 = $0.002224
        # Output: (300  / 1_000_000) * 4.00 = $0.001200
        # Total: ~$0.003424
        cost = _calculate_cost(2780, 300)
        assert 0.003 < cost < 0.004

    def test_cost_is_rounded_to_six_decimal_places(self) -> None:
        cost = _calculate_cost(123, 456)
        # The result should have at most 6 decimal places.
        decimal_str = str(cost).split(".")[-1] if "." in str(cost) else ""
        assert len(decimal_str) <= 6

    def test_output_heavier_than_input(self) -> None:
        # Output is 5x more expensive than input per token.
        input_only = _calculate_cost(1000, 0)
        output_only = _calculate_cost(0, 1000)
        assert output_only > input_only

    def test_cost_increases_with_tokens(self) -> None:
        low = _calculate_cost(100, 100)
        high = _calculate_cost(1000, 1000)
        assert high > low


class TestGenerationResult:
    def test_frozen(self) -> None:
        result = GenerationResult(
            answer="test",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )
        with pytest.raises(Exception):
            result.answer = "mutated"  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        result = GenerationResult(
            answer="Lambda timeout is 15 minutes.",
            input_tokens=500,
            output_tokens=20,
            cost_usd=0.00048,
        )
        assert result.input_tokens == 500
        assert result.output_tokens == 20
        assert result.cost_usd == pytest.approx(0.00048)

.PHONY: install test lint fmt typecheck clean

install:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check src/ tests/

fmt:
	uv run ruff format src/ tests/

typecheck:
	uv run mypy src/

clean:
	rm -rf .venv dist build .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

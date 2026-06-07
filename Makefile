.PHONY: help setup lint fmt typecheck test test-integration up down logs clean

help:
	@echo "setup            Install deps (uv sync) + git hooks (pre-commit)"
	@echo "lint             ruff check + ruff format --check + mypy"
	@echo "fmt              Auto-fix with ruff (check --fix + format)"
	@echo "typecheck        mypy (strict, src/)"
	@echo "test             Run unit tests"
	@echo "test-integration Run tests marked 'integration'"
	@echo "up / down        Start / stop the Qdrant container"
	@echo "logs             Tail Qdrant logs"

setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

fmt:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest

test-integration:
	uv run pytest -m integration

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f qdrant

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage

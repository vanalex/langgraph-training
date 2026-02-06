# Makefile for LangGraph Training Tests

.PHONY: help test test-unit test-integration test-e2e test-coverage test-watch install lint format clean

help:
	@echo "LangGraph Training - Test Commands"
	@echo "===================================="
	@echo "make install          - Install dependencies with uv"
	@echo "make test             - Run all tests"
	@echo "make test-unit        - Run unit tests only"
	@echo "make test-integration - Run integration tests only"
	@echo "make test-e2e         - Run end-to-end tests only"
	@echo "make test-coverage    - Run tests with coverage report"
	@echo "make test-watch       - Run tests in watch mode"
	@echo "make lint             - Run linters"
	@echo "make format           - Format code"
	@echo "make clean            - Clean test artifacts"

install:
	uv sync

test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit -v -m unit

test-integration:
	uv run pytest tests/integration -v -m integration

test-e2e:
	uv run pytest tests/e2e -v -m e2e

test-coverage:
	uv run pytest tests/ -v --cov --cov-report=html --cov-report=term

test-watch:
	uv run pytest-watch tests/ -v

test-fast:
	uv run pytest tests/ -v -m "not slow"

lint:
	uv run ruff check .

format:
	uv run ruff format .

clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

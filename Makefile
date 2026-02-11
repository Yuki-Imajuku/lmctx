.PHONY: lint lint-fix format format-fix typecheck test check check-fix

# Run all checks (lint + typecheck + test)
check: lint format typecheck test

# Run all checks with auto-fix (lint-fix + typecheck + test)
check-fix: lint-fix format-fix typecheck test

# Lint with ruff
lint:
	uv run ruff check

lint-fix:
	uv run ruff check --fix
	uv run ruff format

# Format with ruff
format:
	uv run ruff format --check

format-fix:
	uv run ruff format

# Type check with ty
typecheck:
	uv run ty check

# Run tests
test:
	uv run pytest --cov=lmctx --cov-report=term-missing --cov-report=json:coverage.json --cov-fail-under=90
	uv run python scripts/check_coverage_thresholds.py coverage.json

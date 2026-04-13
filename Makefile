.PHONY: test test-all lint format typecheck docs docker-build benchmark clean precommit install install-dev

# --- Development ---

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# --- Quality ---

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/feaweld/

precommit:
	pre-commit run --all-files

# --- Testing ---

test:
	pytest -m "not slow and not requires_fenics and not requires_calculix and not requires_gmsh and not requires_jax"

test-all:
	pytest

benchmark:
	pytest tests/benchmarks/ --benchmark-only

# --- Documentation ---

docs:
	mkdocs build

docs-serve:
	mkdocs serve

# --- Docker ---

docker-build:
	docker build -t feaweld .

docker-build-fenics:
	docker build -t feaweld-fenics -f Dockerfile.fenics .

# --- Cleanup ---

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache build/ dist/ *.egg-info src/*.egg-info
	rm -rf htmlcov .coverage coverage.xml

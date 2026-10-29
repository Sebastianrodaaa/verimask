.PHONY: install lint test build-failures repro demo readme-table clean

install:
	pip install -e ".[dev]"

lint:
	python3 -m ruff check verimask baselines scripts tests
	python3 -m ruff format --check verimask baselines scripts tests

test:
	python3 -m pytest -q

build-failures:
	python3 scripts/build_failures.py

repro: build-failures
	python3 scripts/run_repro.py --failures tests/mock_data/failures.json

demo:
	pip install matplotlib pillow -q
	python3 scripts/generate_demo_gif.py

readme-table:
	python3 scripts/update_readme_results.py

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

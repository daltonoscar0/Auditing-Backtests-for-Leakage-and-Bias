PYTHON ?= .venv/bin/python
VENV   ?= .venv

.DEFAULT_GOAL := help
.PHONY: help install test eval lint fmt report demo clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-10s\033[0m %s\n", $$1, $$2}'

$(VENV):
	uv venv --python 3.12 $(VENV)

install: $(VENV)  ## Create the virtualenv and install the package with dev extras
	uv pip install --python $(VENV) -e ".[dev]"

test:  ## Run the test suite
	$(PYTHON) -m pytest -q

eval:  ## Score the auditor against the planted-bug fixtures
	$(PYTHON) eval/run_eval.py

lint:  ## Check formatting and lint rules
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

fmt:  ## Apply formatting and safe lint fixes
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

report:  ## Write a markdown report for the mixed fixture to build/mixed-report.md
	@mkdir -p build
	PYTHONPATH=src $(PYTHON) -m hindsight.cli audit fixtures/mixed --out build/mixed-report.md

demo:  ## Print a markdown report for the mixed fixture
	@PYTHONPATH=src $(PYTHON) -m hindsight.cli audit fixtures/mixed

clean:  ## Remove build artefacts and caches
	rm -rf build .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

record:  ## Re-record the model-pass cassette against the live API (needs credentials)
	PYTHONPATH=src $(PYTHON) -m hindsight.cli audit fixtures/llm_only --llm \
		--llm-record fixtures/llm_only/cassette.json

# fair-prefill dev tasks.
#
# `make check` is what CI runs and what to run before pushing. It deliberately
# does NOT cover the scheduler tests, which need a real vLLM -- use
# `make test-container` for those.

PY ?= python3
CONTAINER ?= qw38-vllm
MOUNT ?= /fair-prefill

.PHONY: help lint fmt test check test-container qualname

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

lint: ## ruff check + format check
	ruff check .
	ruff format --check .

fmt: ## apply ruff formatting and autofixes
	ruff check --fix .
	ruff format .

test: ## pytest (vLLM-dependent tests skip when vLLM is absent)
	$(PY) -m pytest -q

check: lint test ## what CI runs

test-container: ## run the full suite inside the serving image, vLLM present
	docker exec -e PYTHONPATH=$(MOUNT) $(CONTAINER) \
	  $(PY) -m pytest -q $(MOUNT)/tests

qualname: ## print the --scheduler-cls value to configure
	@$(PY) -c "import fair_prefill; print(fair_prefill.SCHEDULER_QUALNAME)"

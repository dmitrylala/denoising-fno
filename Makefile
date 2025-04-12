.PHONY: help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-17s\033[0m %s\n", $$1, $$2}'

clean:  ## Remove generated files, caches and build files
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo|.ruff_cache|.pytest_cache|.ipynb_checkpoints)" | xargs rm -rf
	find . | grep -E "(egg-info|build|dist)" | xargs rm -rf

install:  ## Init venv, install all dependencies
	uv sync

test:  ## Run tests
	uv run pytest tests/*.py

fmt:  ## Format and lint
	uv run ruff format . && uv run ruff check . --fix

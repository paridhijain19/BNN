# BBN-JAX developer convenience targets
# Usage: make <target>

PY ?= python

.PHONY: help install install-dev test lint format data train fisher benchmark figures clean

help:
	@echo "Targets:"
	@echo "  install      Install runtime package (editable)"
	@echo "  install-dev  Install with dev + optional extras"
	@echo "  test         Run unit tests"
	@echo "  lint         Run ruff + mypy"
	@echo "  format       Run black + ruff --fix"
	@echo "  data         Generate a small mock training dataset"
	@echo "  train        Train baseline MLP emulator"
	@echo "  fisher       Run Fisher forecast demo"
	@echo "  benchmark    Run LINX-vs-emulator benchmark"
	@echo "  figures      Produce publication figures"
	@echo "  clean        Remove caches and generated artifacts"

install:
	$(PY) -m pip install -e .

install-dev:
	$(PY) -m pip install -e ".[dev,all]"

test:
	$(PY) -m pytest

lint:
	ruff check src tests scripts
	mypy src

format:
	black src tests scripts
	ruff check --fix src tests scripts

data:
	$(PY) scripts/00_generate_data.py --config configs/data_gen.yaml

train:
	$(PY) scripts/01_train_mlp.py --config configs/train_mlp.yaml

fisher:
	$(PY) scripts/06_fisher_forecast.py

benchmark:
	$(PY) scripts/07_benchmark.py

figures:
	$(PY) scripts/08_make_figures.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ build dist *.egg-info
	rm -rf artifacts runs outputs

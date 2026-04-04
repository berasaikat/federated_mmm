.PHONY: install test lint format generate train report

install:
	pip install -e ".[dev]"

generate:
	python run.py generate-data --config config/global.yaml

train:
	python run.py train --config config/global.yaml

validate:
	python run.py validate --config config/global.yaml

visualize:
	python run.py visualize --config config/global.yaml

report:
	python run.py report logs/exp_001 --config config/global.yaml

test-fast:
	python tests/test_regression_math.py

test-smoke:
	python tests/run_smoke_tests.py

test-integration:
	python tests/test_integration_full_pipeline.py

lint:
	mypy participants/ aggregator/ llm_prior/ privacy/ --ignore-missing-imports

format:
	black .
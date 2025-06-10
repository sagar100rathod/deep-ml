.PHONY: install lint test format

install:
	poetry install

lint:
	poetry run black deepml tests
	poetry run isort deepml tests

format:
	poetry run black deepml tests notebooks
	poetry run isort deepml tests

test:
	poetry run pytest tests

check:
	poetry run black deepml tests
	poetry run isort deepml tests
	poetry run pre-commit run --all-files

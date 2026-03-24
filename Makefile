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

.PHONY: docs docs-api docs-build

docs-api:
	poetry run sphinx-apidoc -f -o docs/source/api/ ./deepml/

docs-build:
	cd docs && poetry run make html

docs: docs-api docs-build

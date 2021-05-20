PHONY: setup the work environment
setup:
	cp hooks/commit-msg .git/hooks/
	pre-commit install


install:
	poetry install

test:
	poetry run mypy certifia
	poetry run pytest tests --cov=certifia  --cov-fail-under=85

PHONY: setup the work environment
setup:
	cp hooks/commit-msg .git/hooks/
	poetry install
	pre-commit install


install:
	poetry install

test:
	poetry run mypy certifia
	poetry run pytest tests --cov=certifia  --cov-fail-under=80

run: 
	export FLASK_APP=certifia/main.py
	flask run 
PHONY: setup the work environment
setup:
	cp hooks/commit-msg .git/hooks/
	poetry install
	pre-commit install


install:
	poetry install

test:
	poetry run mypy app
	poetry run pytest tests --cov=app  --cov-fail-under=80

run:
	flask run


build:
	docker build -t certifia .

serve:
	docker run -p 5000:80 certifia

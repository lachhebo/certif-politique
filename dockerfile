FROM tiangolo/uwsgi-nginx-flask:python3.8


RUN apt-get update && apt-get -y upgrade


RUN apt-get install curl  ca-certificates -y

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Install dependencies
COPY . /app

RUN poetry install --no-dev

ENV STATIC_PATH /app/templates

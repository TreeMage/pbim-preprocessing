FROM python:3.10-slim as builder
ENV POETRY_HOME=/opt/poetry
RUN apt update && apt install -y curl
RUN curl -sSL https://install.python-poetry.org/ | python -
ENV PATH="/opt/poetry/bin:${PATH}"
RUN chmod o+rx /opt/poetry/bin/poetry

FROM builder as base
COPY . /app
WORKDIR /app
RUN chmod o+rx scripts/preprocess.sh
RUN poetry install --no-dev



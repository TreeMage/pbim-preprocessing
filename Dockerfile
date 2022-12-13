FROM python:3.10-slim as base
ENV PYTHONUNBUFFERED=1
WORKDIR /app

FROM base as builder
ENV POETRY_HOME=/opt/poetry
RUN apt update && apt install -y curl vim
RUN curl -sSL https://install.python-poetry.org/ | python -
ENV PATH="/opt/poetry/bin:${PATH}"
RUN chmod o+rx /opt/poetry/bin/poetry
RUN python -m venv /venv
COPY . /app
RUN poetry export -f requirements.txt | /venv/bin/pip install -r /dev/stdin
RUN poetry build && /venv/bin/pip install dist/*.whl

FROM builder as final
COPY --from=builder /venv /venv
RUN chmod o+rx scripts/*
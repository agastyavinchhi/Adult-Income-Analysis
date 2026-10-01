FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

COPY main.py adult.csv ./
COPY tests/ tests/

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "main.py"]
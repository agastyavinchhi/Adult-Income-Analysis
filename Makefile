.PHONY: install run test format lint clean

format:
	uv run black .

lint:
	uv run flake8 .

install:
	uv sync

run:
	uv run python main.py

test:
	uv run pytest -v

clean:
	rm -rf __pycache__ .pytest_cache

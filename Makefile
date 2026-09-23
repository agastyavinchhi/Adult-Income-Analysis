.PHONY: install run test clean

install:
	uv sync

run:
	uv run python main.py

test:
	uv run pytest -v

clean:
	rm -rf __pycache__ .pytest_cache

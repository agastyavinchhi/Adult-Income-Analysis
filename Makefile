NOTEBOOK = main.ipynb

.PHONY: install run test notebook execute clean

install:
	uv sync

run:
	uv run python main.py

test:
	uv run pytest

notebook:
	uv run jupyter notebook $(NOTEBOOK)

execute:
	uv run jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOK)

clean:
	rm -rf .ipynb_checkpoints __pycache__ .pytest_cache

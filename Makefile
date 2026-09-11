PYTHON ?= python3
NOTEBOOK = main.ipynb

.PHONY: install run execute clean

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) -m jupyter notebook $(NOTEBOOK)

execute:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOK)

clean:
	rm -rf .ipynb_checkpoints __pycache__

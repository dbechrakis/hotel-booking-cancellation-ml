# Run the same checks as CI: `make check`. Override the interpreter with `make check PYTHON=python3.12`.
PYTHON ?= python

.PHONY: install check lint evidence test

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m ruff check app.py app src tests scripts --select E4,E7,E9,F

evidence:
	PYTHONPATH=src $(PYTHON) ci/verify_evidence.py

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

check: lint evidence test

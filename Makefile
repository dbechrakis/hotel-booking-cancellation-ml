# Run the same checks as CI: `make check`. Override the interpreter with `make check PYTHON=python3.12`.
PYTHON ?= python

.PHONY: install check lint evidence test serve experiments docker

install:
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

lint:
	$(PYTHON) -m ruff check app.py app src tests scripts --select E4,E7,E9,F

evidence:
	PYTHONPATH=src $(PYTHON) ci/verify_evidence.py

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

check: lint evidence test

serve:
	PYTHONPATH=src $(PYTHON) -m uvicorn hotel_cancellation.api:app --reload

experiments:
	PYTHONPATH=src $(PYTHON) scripts/track_experiments.py

docker:
	docker build -t hotel-cancellation-api .

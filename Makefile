.PHONY: setup generate features ingest train evaluate monitor run-api dashboard test lint

setup:
	pip install -r requirements.txt
	pre-commit install || true

generate:
	python data/seed/generate.py --num-vehicles 20000 --offers-per-enquiry 5

features:
	python pipelines/features/build_features.py

ingest:
	python pipelines/ingest/load.py

train:
	python pipelines/train/train_price_model.py
	python pipelines/train/train_conversion_model.py

evaluate:
	python pipelines/evaluate/evaluate.py

monitor:
	python pipelines/monitor/drift.py
	python pipelines/monitor/performance.py
	python pipelines/monitor/alerts.py

run-api:
	uvicorn app.main:app --reload

dashboard:
	streamlit run dashboard/app.py

test:
	MODEL_SOURCE=mock pytest tests/ -v --tb=short

lint:
	ruff check .
	black --check .

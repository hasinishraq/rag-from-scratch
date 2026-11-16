# Windows without make: run the commands after the tab directly (see README).
CONFIG ?= configs/default.yaml

.PHONY: setup lint test ingest ask eval serve ui web demo docker dataset dataset-eval

setup:
	uv sync
	npm --prefix web install

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright
	npm --prefix web run typecheck

test:
	uv run pytest -q

ingest:
	uv run python scripts/ingest.py --config $(CONFIG)

ask:
	uv run python scripts/ask.py "$(Q)" --config $(CONFIG)

eval:
	uv run python scripts/run_eval.py --config $(CONFIG)

serve:
	uv run uvicorn rag.api:app --reload

ui:
	npm --prefix web run dev

web:
	npm --prefix web run build

demo:
	uv run python scripts/serve.py

docker:
	docker compose up --build

# RepliQA demo corpus (fictional documents, CC BY 4.0): download, index, build the eval set.
dataset:
	uv run python scripts/fetch_repliqa.py
	uv run python scripts/ingest.py --config configs/repliqa.yaml
	uv run python scripts/build_repliqa_eval.py --config configs/repliqa.yaml

dataset-eval:
	uv run python scripts/run_eval.py --config configs/repliqa.yaml --questions evals/repliqa_questions.jsonl --split dev --retrieval-only

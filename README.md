# rag-from-scratch

Question answering over your own PDF, Markdown and HTML files, with citations, written from
scratch in Python (no LangChain / LlamaIndex). It refuses when the documents do not contain the
answer. Answers come from **free model APIs** (Groq, Google Gemini, OpenRouter, …) or, optionally,
a **local Ollama** model. Embeddings and reranking run locally on CPU.

Specification: `rag-implementation-guideline.pdf` (ENG-RAG-001). Change log: `CHANGES.md`.

## Quick start

```bash
uv sync                                   # make setup (Python)
npm --prefix web install                  # web UI packages (Node.js 20+)
cp .env.example .env                      # then paste free API keys (or use the Models page)
uv run python scripts/serve.py            # make demo: builds the UI once, serves UI + API on :8000
```

Or in a container: `docker compose up --build` (see `docs/runbook.md`, "Container").

Open http://127.0.0.1:8000 → **Documents** to add files → **Models** → press **Test** on a provider. Only providers that pass
the connection test are used in Auto mode. Then ask on the **Ask** page.

The first ingest downloads the local embedder and reranker (about 220 MB, once). After that the
app needs no network for retrieval; set `HF_HUB_OFFLINE=1` to forbid model downloads.

### Windows (no `make`)

Every Make target is one command; run the part after the tab in the `Makefile`, e.g.
`uv run pytest -q`, `uv run python scripts/ask.py "your question"`.

## Demo dataset: RepliQA

`make dataset` (or the three commands in the Makefile) downloads 120 PDFs from **RepliQA**
(ServiceNow, CC BY 4.0, newest split `repliqa_4`: Company Policies, Incident Report, Local
Technology and Innovation), indexes them into their own store, and writes
`evals/repliqa_questions.jsonl` (600 questions, 107 unanswerable, dev/test split).

RepliQA's documents are about made-up people and organisations, so the model cannot answer from
memory: a correct answer proves the retrieval and grounding work, and its unanswerable questions
test refusals. Run the app on it with
`uv run python scripts/serve.py --config configs/repliqa.yaml`, and measure retrieval with
`make dataset-eval` (no quota). Files live in `datasets/repliqa/` (git-ignored; attribution in
`datasets/repliqa/ATTRIBUTION.md`); your own corpus in `data/docs/` is not touched.

## Web UI

React 19, Tailwind 4, shadcn-style components on Radix, Sonner toasts, Vaul drawers, cmdk
command menu (Ctrl/⌘ K), NumberFlow counters and Motion animations. Strict black and white with
a dark mode that follows the system (toggle in the sidebar). Every ranked passage shows its
relevance as a percentage bar: solid when the answer cites it, hatched when it was retrieved but
not cited. `#/ask?q=your+question` asks on load. All browser calls go to `/api/...`.

Answers are streamed: the ranked passages appear after the search and the text as it is written,
with **Stop** to cancel. Under each answer: copy, and a right/wrong verdict that is stored on its
trace. Traces can be searched, filtered and opened by link (`#/traces?id=…`); documents can be
filtered and removed.

## Configuring models

| Where | What |
|---|---|
| `.env` or environment | `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY` (names are per provider, editable) |
| Models page | Mode (Auto / Free API only / Ollama only), add/edit/reorder/enable providers, paste keys, limits, Ollama address and model, connection test |
| `configs/default.yaml` | First-run seed of the provider list, and all pipeline defaults |
| `configs/provider_presets.yaml` | "Start from" presets in the Add-provider form |

Keys pasted in the app are stored in `data/store/secrets.json` (owner-only permissions) and are
never shown again. A provider is used only if it is marked **free**; the gateway refuses others.

**Ollama is optional.** It is seeded disabled. To use it: install Ollama, `ollama pull <model>`,
then on the Models page set the address, pick the model, enable it and press Test.

**Model ids change often.** The seeded ids (`gpt-oss-120b`, `gemini-3.5-flash-lite`) come from the
design document. If a test reports *Model not found*, press **Load model list** and pick the exact
id the provider offers.

## Commands

| Make | Direct command |
|---|---|
| `make setup` | `uv sync` |
| `make lint` | `uv run ruff check . && uv run ruff format --check . && uv run pyright` |
| `make test` | `uv run pytest -q` (no network, fake providers) |
| `make ingest` | `uv run python scripts/ingest.py --config configs/default.yaml` |
| `make ask Q="…"` | `uv run python scripts/ask.py "…"` (`--retrieve-only` skips the model) |
| `make eval` | `uv run python scripts/run_eval.py --config configs/default.yaml` |
| `make serve` | `uv run uvicorn rag.api:app --reload` (API only) |
| `make ui` | `npm --prefix web run dev` (UI with hot reload on :5173; needs the API on :8000) |
| `make web` | `npm --prefix web run build` (rebuild the UI served by the API) |
| `make demo` | `uv run python scripts/serve.py` (UI + API on :8000) |

## API

`POST /query`, `POST /query/stream` (server-sent events: `sources`, `delta`, `done` or `error`),
`POST /ingest`, `POST /ingest/jobs`, `GET /sources/{chunk_id}`, `DELETE /documents/{doc_id}`,
`PUT /traces/{id}/feedback`, `GET /health`, `GET /health/live`, `GET /health/ready`,
`GET /metrics` (Prometheus text), `GET/POST /providers`,
`PUT/DELETE /providers/{name}`, `POST /providers/{name}/test`, `POST /providers/test-all`,
`GET /providers/{name}/models`, `GET /ollama/models`, `GET /settings`, `PUT /settings/mode`,
`PUT /settings/embedding`, `GET /documents`, `GET /traces[/{id}]` (`limit`, `offset`, `q`,
`status`, `summary`; total in `X-Total-Count`), `GET /eval/runs`.
Interactive docs at http://127.0.0.1:8000/docs. Errors are `{code, message, request_id}`; quota
exhaustion is HTTP 503 `quota_exhausted`, too many questions at once is 429 `busy`. Every
response carries `X-Request-ID`.

## Environment variables

| Variable | Effect |
|---|---|
| `RAG_CONFIG` | Config file (default `configs/default.yaml`) |
| `RAG_ALLOWED_HOSTS` | Extra host names the API answers to, comma-separated (default: this computer only) |
| `RAG_LOG_FORMAT` | `text` (default) or `json`, one object per line |
| `RAG_LOG_LEVEL` | `INFO` (default), `DEBUG`, `WARNING` |
| `RAG_PRELOAD` | `0` skips loading the local models at start (the first question loads them) |
| `HF_HUB_OFFLINE` | `1` forbids model downloads after the first run |

## Layout

```
src/rag/   config types errors db tokens services pipeline api
           streaming (server-sent events)  ops (health, metrics)  observability  security
           ingest/ loaders clean chunkers ledger runner
           index/  embedder vector_store bm25 chunk_store indexes
           retrieve/ hybrid fusion rerank      generate/ context prompts citations
           llm/    gateway providers ratelimit cache settings secrets ollama healthcheck
web/       React + Vite + Tailwind UI: src/App.tsx, pages/, components/ (served from web/dist)
scripts/   ingest.py ask.py serve.py run_eval.py
evals/     metrics.py judge.py questions.example.jsonl
tests/     unit/ integration/ fixtures/ (fake provider + fake Ollama server, no network)
docs/      runbook.md report.md decisions/
```

# AI Research Navigator

A citation-grounded RAG system over a curated AI-research corpus (arXiv papers,
HuggingFace course chapters, Lil'Log surveys, lab blog posts). This repository is
built milestone by milestone; **M2 delivers the query pipeline** — hybrid retrieval
in Qdrant, deduplicated citations, and a low-confidence refusal path — on top of the
M1 ingestion pipeline.

## Status

- **M0 — Scaffold:** project layout, config, structured logging, tooling, Qdrant
  compose. (done)
- **M1 — Ingestion:** manifest loading, per-content-type parsing/cleaning,
  structure-aware chunking, dense + sparse embeddings, idempotent Qdrant upserts, a
  CLI, and a test suite. (done)
- **M2 — Query pipeline:** query understanding → filter inference, hybrid
  dense+sparse retrieval (RRF, filtered inside Qdrant), Perplexity-style deduplicated
  citations, and a tuned low-confidence refusal. (this milestone)
- M3+ — LangGraph agent layer, evaluation harness. (upcoming)

Design decisions are recorded under [`adr/`](adr/README.md).

## Requirements

- Python >= 3.13 and [uv](https://docs.astral.sh/uv/) (see [ADR-0002](adr/0002-package-and-env-management-uv.md)).
- Docker (for a local Qdrant), or any reachable Qdrant instance.
- The corpus package unpacked at the repo root as `ai-research-navigator-corpus/`
  (it is **not** committed — it is sealed / non-redistributable).

## Setup

```bash
uv sync                       # create the env + install deps from the lockfile
uv run pre-commit install     # optional: enable git hooks
cp .env.example .env          # then edit paths/model as needed
```

The first real ingestion run downloads the dense model (`bge-small`, ~130 MB) and
caches it; later runs are offline. CI/tests use a deterministic offline embedder and
never hit the network (see [ADR-0001](adr/0001-dense-embedding-model.md)).

## Ingesting the corpus

Start Qdrant, then run the pipeline:

```bash
make up                                                # docker compose up -d  (Qdrant on :6333)
uv run python -m research_navigator.ingest validate    # parse+chunk only, no writes
uv run python -m research_navigator.ingest ingest       # parse -> chunk -> embed -> upsert
uv run python -m research_navigator.ingest stats         # counts by content_type/year/tags
```

Other commands:

```bash
# Re-run anytime — unchanged docs write nothing (idempotent; see ADR-0005):
uv run python -m research_navigator.ingest ingest

# Re-ingest a single document (e.g. after editing it):
uv run python -m research_navigator.ingest ingest --doc <doc_id>

# Force a full re-embed, or rebuild the collection from scratch:
uv run python -m research_navigator.ingest ingest --force
uv run python -m research_navigator.ingest reindex
```

All commands print a JSON report to **stdout**; structured logs go to **stderr**, so
output stays pipeable (`... ingest | jq .total_added`). Any per-document error sets a
non-zero exit code while letting the rest of the run complete.

## Querying the corpus

Once the corpus is ingested, ask questions. By default this uses an **offline,
deterministic extractive backend** — no LLM server and no API key required — so a
fresh clone gets grounded, cited answers immediately (see
[ADR-0006](adr/0006-generation-backend-and-generator-abstraction.md)):

```bash
uv run python -m research_navigator.cli.query "What is retrieval-augmented generation?"
uv run python -m research_navigator.cli.query "Compare DPO and KTO" --json
```

For fluent multi-source synthesis, point the OpenAI-compatible backend at a local
OSS model (OSS-first, still keyless):

```bash
ollama serve && ollama pull llama3.1:8b-instruct-q4_K_M   # one-time
export RN_GENERATION__BACKEND=openai
uv run python -m research_navigator.cli.query "How does chain-of-thought prompting work?"
```

Every factual sentence carries an inline `[n]` citation mapping to a deduplicated
sources block (title, authors *et al.*, year, source, section, URL). When retrieval
confidence is below the tuned threshold, the system **refuses** rather than
fabricate; tune it without code via `RN_RETRIEVAL__REFUSAL_MIN_SCORE`
(default `0.30`, see [ADR-0007](adr/0007-refusal-threshold-and-confidence-signal.md)).
The CLI exits non-zero only on a hard generation error — a refusal is a valid outcome.

See [`docs/query-pipeline.md`](docs/query-pipeline.md) for the full M2 architecture.

## Development

```bash
make lint                 # ruff check + ruff format --check + mypy (strict)
make test                 # unit tests
make test-integration     # tests that spin up in-memory Qdrant
uv run pytest --cov       # coverage (gate: >= 70% on src/; M1 is ~90%)
```

See [`docs/ingestion.md`](docs/ingestion.md) and
[`docs/query-pipeline.md`](docs/query-pipeline.md) for the pipeline architectures, and
[`adr/`](adr/README.md) for the decisions behind them.

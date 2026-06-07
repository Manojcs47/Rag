# AI Research Navigator

A citation-grounded RAG system over a curated AI-research corpus (arXiv papers,
HuggingFace course chapters, Lil'Log surveys, lab blog posts). Every factual
sentence carries an inline `[n]` citation linking to the originating source —
or the system **refuses** rather than fabricate. A LangGraph agent layer
routes each query into one of six specialised sub-capabilities, and an
evaluation harness measures retrieval precision/recall, citation faithfulness,
refusal correctness, latency, and approximate token cost across multiple
retrieval configurations.

## Status

| Milestone | Scope | Status |
|---|---|---|
| **M0** — Scaffold | Project layout, config (pydantic-settings), structured logging (structlog), tooling (ruff/mypy/pytest), Docker Compose for Qdrant | done |
| **M1** — Ingestion | Manifest loading, per-content-type parsing/cleaning, structure-aware chunking, dense + sparse embeddings, idempotent Qdrant upserts, CLI, test suite | done |
| **M2** — Query pipeline | Query understanding → filter inference, hybrid dense+sparse retrieval (RRF, filtered inside Qdrant), Perplexity-style deduplicated citations, tuned low-confidence refusal | done |
| **M3** — Agent layer | LangGraph state machine with six routes (concept_explanation, paper_deep_dive, compare_approaches, recent_developments, find_papers, out_of_scope), deterministic router, three typed tools, Mermaid graph visualisation | done |
| **M4** — Evaluation | 48-question golden set across all six routes + held-out OOC, P/R@5 + citation faithfulness (heuristic + optional LLM judge) + refusal correctness + latency + approximate tokens, three-config comparison, JSON + Markdown report | done |
| **M5** — Engineering standards | Continuous: strict mypy, ruff, ≥70% coverage on `src/`, no hardcoded paths/models/thresholds, no silent failure, structured logs only | applied throughout |
| Bonus tracks (B1–B4) | Query UI / Ingestion UI / Reranker / Vernacular | not started |

Design decisions are recorded under [`adr/`](adr/README.md).

---

## Quick start

```bash
# 1. Install uv (https://docs.astral.sh/uv/) if you don't have it.
curl -LsSf https://astral.sh/uv/install.sh | sh    # or: pipx install uv

# 2. Install deps + git hooks.
make setup

# 3. Start Qdrant (Docker).
make up

# 4. Unpack the corpus, complete it (downloads arXiv PDFs + scrapes lab blogs),
#    then validate.
unzip ai-research-navigator-corpus.zip
cd ai-research-navigator-corpus
pip install requests trafilatura html2text      # one-time, for the completion script
python3 complete_corpus.py                       # ~3 minutes (polite arXiv delays)
cd ..

# 5. Ingest the corpus into Qdrant.
uv run python -m research_navigator.ingest ingest

# 6. Ask a question, run the full agent layer, or run the eval harness.
uv run python -m research_navigator.cli.query  "What is retrieval-augmented generation?"
uv run python -m research_navigator.cli.agent  "Compare DPO and KTO"
make eval                                        # writes eval/report.{json,md}
```

A fresh clone is reviewer-ready in well under five minutes once the corpus has
been completed.

---

## The system at a glance

```
                                      query
                                        │
                                        v
                         ┌─────── LangGraph Router (M3) ───────┐
                         │  regex-based intent classification  │
                         │   + hint extraction (paper, targets)│
                         └─────────────────┬───────────────────┘
                                           │ conditional edge
        ┌─────────────────┬────────────────┼─────────────────┬─────────────────┐
        v                 v                v                 v                 v
 concept_explanation  paper_deep_dive  compare_approaches  recent_developments  find_papers
        │                 │                │                 │                 │
        │  M2 directly    │  +doc_id       │  per-target     │  date_math +    │  metadata-only
        │                 │   filter       │  retrieval,     │  forced year    │  ranking, no
        │                 │                │  combined       │  floor          │  LLM generation
        └──────┬──────────┴────────────────┴────────┬────────┴────────┬────────┘
               │                                    │                 │
               v                                    v                 v
       ┌────────────────────── M2 Query pipeline ──────────────────────┐
       │  analyze_query  →  build_filter  →  HybridRetriever            │
       │      (filter inference)     (server-side Qdrant filter)        │
       │      dense (bge-small) + sparse (TF + Qdrant IDF) ── RRF ──┐   │
       │                                                            │   │
       │                                          dense-only        │   │
       │                                          confidence pass ──┤   │
       │                                                            v   │
       │                              refusal gate (cosine < threshold) │
       │                                          │                     │
       │                                          v                     │
       │  build_citations (dedup by doc_id, number by score)             │
       │                                          │                     │
       │                                          v                     │
       │  Generator:  extractive (default)  |  OpenAI-compatible         │
       │                                          │                     │
       │                                          v                     │
       │  validate [n] markers; unattributed → refuse                    │
       └──────────────────────────────────┬──────────────────────────────┘
                                          │
                                          v
                            Answer (text + numbered citations)
                              or  REFUSED  or  ERROR
```

**Mandated stack:** Python ≥ 3.13, Qdrant (vector store + metadata filtering),
LangGraph (agent orchestration). Plus: sentence-transformers (`bge-small-en-v1.5`)
for dense embeddings, a hashed TF sparse vectoriser with Qdrant's IDF modifier
(== BM25, ADR-0004), an OpenAI-compatible LLM client for optional fluent
synthesis, `pydantic-settings` for config, `structlog` for logs, `uv` for
packaging, `ruff` + `mypy --strict` for hygiene, `pytest` for testing,
Docker Compose for Qdrant.

---

## Requirements

- **Python ≥ 3.13** and [**uv**](https://docs.astral.sh/uv/) (see [ADR-0002](adr/0002-package-and-env-management-uv.md)).
- **Docker** for a local Qdrant, or any reachable Qdrant instance.
- The **corpus package** unpacked at the repo root as `ai-research-navigator-corpus/`.
  It is *not* committed (sealed / non-redistributable). The `complete_corpus.py`
  script in the corpus directory downloads the 30 arXiv PDFs and scrapes the 3
  lab-blog markdowns the manifest references — see the corpus README inside the
  unzipped folder for details.
- *(Optional, only for M2 fluent synthesis or the M4 LLM judge)*: a running
  OpenAI-compatible chat endpoint such as [Ollama](https://ollama.com)
  serving `llama3.1:8b-instruct-q4_K_M`. The defaults work entirely offline.

---

## Setup

```bash
uv sync                       # create the env + install deps from the lockfile
uv run pre-commit install     # optional: enable git hooks
cp .env.example .env          # then edit paths/model as needed
```

The first real ingestion run downloads the dense embedding model
(`BAAI/bge-small-en-v1.5`, ~130 MB) and caches it in your HuggingFace cache
directory; subsequent runs are offline. CI/tests use a deterministic offline
hashing embedder (`RN_USE_OFFLINE_EMBEDDER=1`) and never hit the network — see
[ADR-0001](adr/0001-dense-embedding-model.md).

---

## Corpus setup

The corpus is shipped as `ai-research-navigator-corpus.zip`. Unpack it at the
repository root, then complete the missing arXiv PDFs and lab-blog markdowns:

```bash
unzip ai-research-navigator-corpus.zip
cd ai-research-navigator-corpus
pip install requests trafilatura html2text
python3 complete_corpus.py        # ~3 minutes; polite arXiv delays
cd ..
```

The script is idempotent — re-running it does nothing once the corpus is
complete. After it finishes, `documents/` should contain 30 arXiv PDFs +
12 HuggingFace markdown chapters + 5 Lil'Log markdowns + 3 lab-blog markdowns
= 50 documents matching `manifest.json`.

The `RN_CORPUS_DIR` and `RN_MANIFEST_PATH` env vars let you keep the corpus
anywhere; the codebase treats the corpus as a **variable, not a constant**,
so the same code works against a different sealed corpus when the client
swaps it in.

---

## M1 — Ingestion pipeline

Turns the sealed corpus into retrievable, citable chunks in Qdrant.

```
manifest.json
     │
     v
[manifest]   load + validate (pydantic)                ingest/manifest.py
     │
     v
[parse]      PDF (PyMuPDF) | Markdown (regex)          ingest/parse.py
     │         - strip MDX/anchors/comments/nav cruft
     │         - isolate abstract (papers)
     │         - split off References (kept, not retrieved)
     │         - build a section tree with heading hierarchy
     v
[chunk]      section-bounded, content-type budgets     ingest/chunk.py
     │         - abstract = own chunk; code atomic
     │         - token-bounded packing + overlap
     │         - content_hash + deterministic point_id (UUIDv5)
     v
[embed]      dense (sentence-transformers | offline)   ingest/embed.py
     │       sparse (TF over hashed vocab; IDF in Qdrant)
     v
[store]      idempotent upsert into Qdrant             ingest/qdrant_store.py
                 - named dense (cosine) + sparse (IDF) vectors
                 - payload = all manifest fields + per-chunk fields
                 - payload indexes for M2 filtering
```

### Commands

```bash
# Start Qdrant (REST on :6333, dashboard at http://localhost:6333/dashboard).
make up

# Parse + chunk every document without writing to Qdrant.
uv run python -m research_navigator.ingest validate

# Full ingest: parse → chunk → embed → upsert.
uv run python -m research_navigator.ingest ingest

# Counts by content_type / year / tags.
uv run python -m research_navigator.ingest stats

# Re-ingest a single document (e.g. after editing it on disk).
uv run python -m research_navigator.ingest ingest --doc <doc_id>

# Force a full re-embed and upsert (ignore idempotency cache).
uv run python -m research_navigator.ingest ingest --force

# Drop the collection and rebuild from scratch.
uv run python -m research_navigator.ingest reindex
```

JSON reports go to **stdout**; structured logs go to **stderr**, so you can
pipe output cleanly: `... ingest | jq .total_added`. Any per-document error
sets a non-zero exit code while letting the rest of the run complete (no
silent failure — M5).

### Idempotency contract

A chunk's Qdrant id is `uuid5(NAMESPACE, "{doc_id}:{chunk_index}:{content_hash}")`.
On each run, per document, the pipeline computes the desired id set, reads the
existing id set from Qdrant, then upserts only new ids, deletes only stale
ids, and skips the document entirely when the sets match.

- Re-ingesting an unchanged corpus writes nothing.
- Editing one document re-embeds/upserts only its changed chunks and deletes
  only its orphaned chunks; every other document is untouched.

Both properties are covered in `tests/test_pipeline_integration.py`.
Full details: [`docs/ingestion.md`](docs/ingestion.md).

---

## M2 — Query pipeline

Turns a learner's question into a grounded, citation-carrying answer — or a
graceful refusal.

```
question
   │
   v
[understand]  free text -> inferable metadata filters     retrieve/query.py
   │            - year bounds ("recent" / "since YYYY")
   │            - tags from the corpus vocabulary
   │            - content_type / foundational hints
   v
[filter]      QueryAnalysis -> Qdrant Filter              retrieve/filters.py
   v
[retrieve]    hybrid dense+sparse, RRF-fused, filtered    retrieve/hybrid.py
   │            server-side; + a dense-only pass for the
   │            confidence signal (ADR-0007)
   │            -- relax tags once if a filter empties result
   v
[gate]        refuse if confidence < threshold or empty   generate/pipeline.py
   v
[cite]        dedup chunks -> numbered citation block     generate/citations.py
   │            (title, authors et al., year, source, section, URL)
   v
[generate]    extractive (default) | OpenAI-compatible    generate/generator.py
   v
[validate]    strip out-of-range [n]; no attribution ->   generate/pipeline.py
              refuse. Never let an unattributed claim stand.
```

### Commands

```bash
# Offline / deterministic default (no key, no LLM server): extractive backend.
uv run python -m research_navigator.cli.query "What is retrieval-augmented generation?"

# JSON output (status, citations, confidence) for scripting / eval.
uv run python -m research_navigator.cli.query "Compare DPO and KTO" --json

# Fluent synthesis via a local OSS model (OSS-first, still keyless):
ollama serve && ollama pull llama3.1:8b-instruct-q4_K_M    # one-time
export RN_GENERATION__BACKEND=openai
uv run python -m research_navigator.cli.query "How does chain-of-thought prompting work?"
```

Tune the refusal floor against your embedding model without touching code:
`export RN_RETRIEVAL__REFUSAL_MIN_SCORE=0.35`. Default `0.30` is the
conservative floor calibrated for `bge-small` with its query-instruction
prefix (ADR-0007). The CLI exits non-zero only on a hard generation error —
a refusal is a valid outcome.

### The citation contract (M2 acceptance)

The pipeline enforces the contract **structurally**, not by trusting the
generator:

1. **Refuse on low confidence.** Dense-cosine confidence below
   `RN_RETRIEVAL__REFUSAL_MIN_SCORE` or empty retrieval → the fixed refusal
   message (ADR-0007).
2. **Cite only real, retrieved sources.** Citations are built *from the
   retrieved chunks*, deduplicated and numbered; the generator is handed
   exactly those numbered sources.
3. **Validate every marker.** After generation, `[n]` markers are checked
   against the real citation set — out-of-range markers (a model inventing
   `[9]`) are stripped.
4. **No unattributed answer.** If validation leaves an answer with no valid
   `[n]`, it is converted to a refusal.

So a fabricated citation cannot survive, and a hallucinated answer degrades
to a refusal — independent of which generation backend is configured.

Full details: [`docs/query-pipeline.md`](docs/query-pipeline.md).

---

## M3 — LangGraph agent layer

Routes each query into one of six specialised sub-capabilities and runs that
single node. State is explicit, pydantic-typed, JSON-serialisable.

### The six routes

| Route | When the router picks it | What the node does |
|---|---|---|
| `concept_explanation` | "what is", "how does", "explain" — the default | Calls the M2 pipeline directly (multi-source synthesis) |
| `paper_deep_dive` | Query references a specific paper (arXiv id, distinctive title word, method alias) | Filters retrieval to that document's chunks |
| `compare_approaches` | "compare", "vs", "difference between" + 2+ identifiable targets | Retrieves per target, combines sources, generates a comparison |
| `recent_developments` | "recent", "latest", "state-of-the-art" | Uses the `date_math` tool to derive a year floor, forces recency |
| `find_papers` | "reading list", "recommend papers", "where to start" | Metadata-only ranking via `corpus_metadata_lookup` + `rank_papers_for_reading_list` — no LLM generation |
| `out_of_scope` | Off-topic regex matches + no in-scope signal | Polite decline (fallback node) |

### Tools

Three typed Python helpers, invoked from nodes and recorded as
`ToolInvocation` entries on `AgentState.tool_log` for traceability:

- **`corpus_metadata_lookup`** — filters the manifest by tags / content_type /
  is_foundational / year / doc_id.
- **`date_math`** — computes a year floor `months_back` months before today.
- **`rank_papers_for_reading_list`** — ranks candidates on
  `(is_foundational, tag_match_count, year)`. `citation_count` is null for
  every doc in the sealed corpus (Roadmap §0 finding #4), so we rank without
  it — documented in [ADR-0008](adr/0008-agentic-routing-with-langgraph.md).

### Commands

```bash
# A specific paper:
uv run python -m research_navigator.cli.agent "Tell me about Llama 3"

# A comparison:
uv run python -m research_navigator.cli.agent "Compare DPO and KTO"

# Recent developments:
uv run python -m research_navigator.cli.agent "Recent work on agents"

# A reading list:
uv run python -m research_navigator.cli.agent "Recommend seminal papers on alignment"

# Out of scope:
uv run python -m research_navigator.cli.agent "What's the weather today?"

# Full JSON output (route, hints, tool_log, full answer):
uv run python -m research_navigator.cli.agent "Compare DPO and KTO" --json

# Render the LangGraph state machine to a Mermaid diagram (GitHub renders it).
uv run python -m research_navigator.cli.agent --visualize docs/agent-graph.md
```

Each route is exercised by ≥3 deterministic test queries in
`tests/test_agent_router.py`. Full details:
[`docs/agent-layer.md`](docs/agent-layer.md).

---

## M4 — Evaluation harness

Reproducible measurement of the system's retrieval P/R, citation faithfulness,
refusal correctness, latency, and approximate token cost, across **three**
retrieval configurations.

### Configurations compared

| ID | Name | Sparse branch | Metadata filters |
|----|------|---------------|------------------|
| A | `hybrid_with_filters` (baseline) | yes | yes |
| B | `dense_only_with_filters` | no | yes |
| C | `hybrid_no_filters` | yes | no |

A vs B isolates the contribution of the sparse branch (ADR-0004).
A vs C isolates the contribution of server-side metadata filtering.

### Metrics

- **Retrieval P/R @ 5** computed on the doc_ids that appear in the **final
  answer's citations** (i.e. end-to-end, what the learner sees), against
  the golden-set `expected_doc_ids`.
- **Citation faithfulness.** Heuristic by default (coverage of `[n]` markers
  + structural marker support, both in [0,1], averaged). Opt-in LLM judge via
  `--llm-judge` runs the same explicit rubric through any OpenAI-compatible
  model — see [ADR-0009](adr/0009-evaluation-harness.md).
- **Refusal correctness** on a held-out OOC set (8 questions: 5 clearly
  off-topic + 3 in-domain phrasings about papers not in the corpus).
- **Latency** p50, p95, mean, max per configuration.
- **Approximate tokens** via the 4-chars-per-token heuristic, comparable
  across configs (honest limitation called out in the report).

### Golden set

`eval/golden_set.jsonl` — 40 in-corpus + 8 out-of-corpus = 48 questions,
≥6 per route, every route covered.

### Commands

```bash
# Default: heuristic judge, three-config comparison. Writes eval/report.{json,md}.
make eval

# Custom golden set / output directory.
uv run python -m research_navigator.cli.eval --golden eval/golden_set.jsonl --out eval/

# Opt in to the LLM judge (any OpenAI-compatible endpoint).
export RN_GENERATION__BACKEND=openai
export RN_GENERATION__LLM_BASE_URL=http://localhost:11434/v1
export RN_GENERATION__LLM_MODEL=llama3.1:8b-instruct-q4_K_M
uv run python -m research_navigator.cli.eval --llm-judge

# Read the one-page report.
cat eval/report.md
```

The report includes a headline metrics table, latency + cost table, a
per-route breakdown, comparative findings (A vs B, A vs C deltas), and an
"Honest limitations" section that names every approximation by its number.
Full details: [`docs/evaluation.md`](docs/evaluation.md).

---

## Project layout

```
research-navigator/
├── README.md                        # this file
├── ARCHITECTURE.md                  # system diagram + design rationale (companion)
├── docker-compose.yml               # local Qdrant on :6333
├── Makefile                         # setup / lint / test / eval / up / down
├── pyproject.toml                   # uv-managed deps + ruff / mypy / pytest config
├── uv.lock                          # pinned dependency graph (committed)
├── .pre-commit-config.yaml          # ruff, mypy, detect-secrets
├── .env.example                     # all env vars, with comments
├── adr/                             # >= 9 architecture decision records
│   ├── README.md
│   ├── 0001-dense-embedding-model.md
│   ├── 0002-package-and-env-management-uv.md
│   ├── 0003-chunking-strategy.md
│   ├── 0004-sparse-retrieval-and-fusion.md
│   ├── 0005-deterministic-point-ids-and-idempotency.md
│   ├── 0006-generation-backend-and-generator-abstraction.md
│   ├── 0007-refusal-threshold-and-confidence-signal.md
│   ├── 0008-agentic-routing-with-langgraph.md
│   └── 0009-evaluation-harness.md
├── docs/
│   ├── ingestion.md                 # M1 walk-through
│   ├── query-pipeline.md            # M2 walk-through
│   ├── agent-layer.md               # M3 walk-through
│   ├── agent-graph.md               # auto-generated Mermaid diagram
│   └── evaluation.md                # M4 walk-through
├── src/research_navigator/
│   ├── __init__.py
│   ├── config.py                    # pydantic-settings; all thresholds/models/paths
│   ├── logging.py                   # structlog setup (stderr; JSON optional)
│   ├── ingest/                      # M1
│   │   ├── manifest.py              # load + validate manifest.json
│   │   ├── parse.py                 # PDF + Markdown parsers
│   │   ├── chunk.py                 # section-bounded chunking
│   │   ├── embed.py                 # dense + sparse embedders
│   │   ├── qdrant_store.py          # collection schema + upserts
│   │   └── pipeline.py              # orchestration + idempotency diff
│   ├── retrieve/                    # M2 (retrieval)
│   │   ├── query.py                 # free text -> QueryAnalysis (filters only)
│   │   ├── filters.py               # QueryAnalysis -> Qdrant Filter
│   │   └── hybrid.py                # HybridRetriever + Retriever Protocol
│   ├── generate/                    # M2 (generation)
│   │   ├── prompt.py                # system prompt + REFUSAL_TEXT
│   │   ├── citations.py             # dedup + numbered citation block
│   │   ├── generator.py             # Extractive + OpenAI-compatible backends
│   │   └── pipeline.py              # QueryEngine (refuse/cite/generate/validate)
│   ├── agents/                      # M3
│   │   ├── state.py                 # AgentRoute, RoutingHints, AgentState
│   │   ├── router.py                # deterministic regex classifier
│   │   ├── tools.py                 # corpus_metadata_lookup, date_math, ranker
│   │   ├── nodes.py                 # 6 node functions + Fallback
│   │   └── graph.py                 # LangGraph wiring + Mermaid renderer
│   ├── eval/                        # M4
│   │   ├── golden_set.py            # JSONL loader + pydantic schema
│   │   ├── metrics.py               # P/R@k, latency percentiles, tokens, faithfulness
│   │   ├── judge.py                 # LLM-as-judge (+ heuristic fallback)
│   │   ├── retrievers.py            # DenseOnlyRetriever (ablation)
│   │   ├── runner.py                # orchestrate configs × golden set
│   │   └── report.py                # render JSON + Markdown report
│   └── cli/                         # thin argparse entrypoints (no business logic)
│       ├── ingest.py                # rn-ingest
│       ├── query.py                 # rn-query
│       ├── agent.py                 # rn-agent
│       └── eval.py                  # rn-eval
├── tests/                           # unit + integration (>= 70% coverage on src/)
│   ├── conftest.py                  # tiny self-contained fixture corpus
│   ├── test_smoke.py
│   ├── test_manifest.py
│   ├── test_parse.py
│   ├── test_chunk.py
│   ├── test_embed.py
│   ├── test_qdrant_store.py
│   ├── test_pipeline_integration.py
│   ├── test_cli.py
│   ├── test_query_understanding.py
│   ├── test_filters.py
│   ├── test_hybrid.py
│   ├── test_citations.py
│   ├── test_generator.py
│   ├── test_pipeline.py
│   ├── test_agent_router.py
│   ├── test_agent_tools.py
│   ├── test_agent_graph.py
│   ├── test_eval_metrics.py
│   ├── test_eval_golden_set.py
│   └── test_eval_runner.py
└── eval/
    ├── golden_set.jsonl             # 40 in-corpus + 8 OOC = 48 questions
    ├── report.json                  # generated by `make eval`
    └── report.md                    # generated by `make eval`
```

---

## Configuration reference

Everything tunable lives in `research_navigator.config.Settings`. Read from
the environment via `pydantic-settings` (prefix `RN_`, `.env` honoured,
nested params via the `__` delimiter). Nothing in `src/` hardcodes a path,
model, or threshold.

| Variable | Default | Purpose |
|---|---|---|
| `RN_CORPUS_DIR` | `ai-research-navigator-corpus` | Root of the unpacked corpus |
| `RN_MANIFEST_PATH` | `ai-research-navigator-corpus/manifest.json` | Manifest location |
| `RN_QDRANT_URL` | `http://localhost:6333` | Qdrant REST endpoint (`:memory:` for in-process) |
| `RN_QDRANT_COLLECTION` | `research_navigator` | Collection name |
| `RN_DENSE_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence-transformers model id |
| `RN_DENSE_DIM` | `384` | Dense embedding dim |
| `RN_DENSE_DEVICE` | `cpu` | `cuda` if GPU available |
| `RN_USE_OFFLINE_EMBEDDER` | `false` | Deterministic hashing embedder (CI/tests) |
| `RN_QUERY_PREFIX` | `Represent this sentence for searching relevant passages: ` | bge-required query instruction |
| `RN_SPARSE_VOCAB_SIZE` | `262144` | Hashed sparse vocab buckets |
| `RN_CHUNK__ARXIV_PAPER__MAX_TOKENS` | `512` | Token budget per arXiv chunk (similar nested vars for other content types) |
| `RN_RETRIEVAL__TOP_K` | `5` | Deduped sources surfaced to generation |
| `RN_RETRIEVAL__CANDIDATE_K` | `20` | Chunks fused before dedup |
| `RN_RETRIEVAL__PREFETCH_LIMIT` | `40` | Candidates per branch before RRF |
| `RN_RETRIEVAL__REFUSAL_MIN_SCORE` | `0.30` | Dense-cosine confidence floor — below it, refuse |
| `RN_RETRIEVAL__RECENT_YEAR_FLOOR` | `2024` | Year a bare "recent" maps to |
| `RN_GENERATION__BACKEND` | `extractive` | `extractive` (default) or `openai` |
| `RN_GENERATION__MAX_SENTENCES` | `6` | Cap on extractive answer length |
| `RN_GENERATION__LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible chat endpoint |
| `RN_GENERATION__LLM_MODEL` | `llama3.1:8b-instruct-q4_K_M` | Model id passed to the chat API |
| `RN_GENERATION__LLM_API_KEY` | `None` | Bearer token (omit for keyless local servers) |
| `RN_GENERATION__LLM_TEMPERATURE` | `0.0` | Sampling temperature (0.0 for reproducibility) |
| `RN_SEED` | `42` | Global seed for any stochastic step |
| `RN_LOG_LEVEL` | `INFO` | Root log level |
| `RN_JSON_LOGS` | `false` | `true` → JSON logs to stderr (keeps stdout pipeable) |

See `.env.example` for a copy-paste-ready template with comments.

---

## Development

```bash
make setup                 # uv sync + pre-commit install
make lint                  # ruff check + ruff format --check + mypy (strict)
make fmt                   # auto-fix with ruff
make test                  # unit tests
make test-integration      # tests that spin up in-memory Qdrant
uv run pytest --cov        # coverage report (gate: >= 70% on src/)

# Service control
make up                    # docker compose up -d  (Qdrant on :6333)
make down                  # stop (data persists in the named volume)
make logs                  # tail Qdrant logs

# Evaluation
make eval                  # writes eval/report.{json,md}
```

**Engineering standards (M5, enforced throughout):**

- Clear `src/` layout; modular `ingest` / `retrieve` / `generate` / `agents` /
  `eval` packages; no business logic in CLI entrypoints.
- Strict `mypy`; no `Any` in public interfaces without justification.
- `pydantic-settings` for all config; no hardcoded paths, models, or thresholds.
- `structlog` everywhere; no `print` outside CLI user-facing output.
- Unit tests for parsing, chunking, filter inference, citation rendering,
  refusal, router, tools, eval metrics. Integration tests for the ingest
  CLI, the LangGraph graph, and the eval runner. ≥70% coverage on `src/`.
- Reproducibility: fixed seeds, locked deps committed, `docker compose up`
  brings the stack up.
- No silent failure: every degraded component logs clearly and either
  recovers or fails loudly. `except: pass` is treated as a bug.

---

## Architecture decisions (ADRs)

Lightweight [MADR](https://adr.github.io/madr/): Context · Decision ·
Consequences · Alternatives. Immutable once `Accepted`.

| ADR | Title | Status |
|-----|-------|--------|
| [0001](adr/0001-dense-embedding-model.md) | Local open-source dense embeddings via sentence-transformers (`bge-small-en-v1.5`) | Accepted |
| [0002](adr/0002-package-and-env-management-uv.md) | `uv` for packaging and environment management | Accepted |
| [0003](adr/0003-chunking-strategy.md) | Section-bounded, content-type-aware chunking | Accepted |
| [0004](adr/0004-sparse-retrieval-and-fusion.md) | TF sparse vectors + Qdrant IDF (BM25) with RRF fusion | Accepted |
| [0005](adr/0005-deterministic-point-ids-and-idempotency.md) | Deterministic UUIDv5 point IDs for idempotent ingestion | Accepted |
| [0006](adr/0006-generation-backend-and-generator-abstraction.md) | Extractive-by-default generation behind an OpenAI-compatible backend | Accepted |
| [0007](adr/0007-refusal-threshold-and-confidence-signal.md) | Refusal on a dense-cosine confidence signal, not the fused RRF score | Accepted |
| [0008](adr/0008-agentic-routing-with-langgraph.md) | Agentic routing with LangGraph: deterministic router, typed tools, metadata-only FindPapers | Accepted |
| [0009](adr/0009-evaluation-harness.md) | Multi-configuration eval harness with end-to-end metrics and a dual-mode faithfulness judge | Accepted |

---

## Demo walkthrough (10–15 minutes)

A suggested path for a reviewer or a recorded demo:

1. **Setup (≤ 1 min).** `make setup && make up`, then show `make lint` passes.
2. **Corpus + ingestion (≤ 2 min).** Show `manifest.json`, run
   `uv run python -m research_navigator.ingest validate` (no writes), then
   `... ingest stats` to show per-content-type / per-year / per-tag counts.
   Re-run `ingest` to demonstrate idempotency (`total_writes: 0`).
3. **M2 — query pipeline (≤ 3 min).** Run two questions:
   - `What is retrieval-augmented generation?` → grounded answer + sources.
   - `What's the capital of Mars?` → refusal (the dense-cosine gate fires).
   Show the JSON output (`--json`) to make the citation contract visible.
4. **M3 — agent layer (≤ 4 min).** One query per route:
   - `Tell me about Llama 3` → `paper_deep_dive`.
   - `Compare DPO and KTO` → `compare_approaches`.
   - `Recent work on agents` → `recent_developments`.
   - `Recommend seminal papers on alignment` → `find_papers` (metadata-only).
   - `What's the weather today?` → `out_of_scope`.
   Open `docs/agent-graph.md` for the Mermaid diagram.
5. **M4 — eval harness (≤ 3 min).** `make eval`, then `cat eval/report.md`.
   Walk through the headline table, the per-route breakdown, the A-vs-B and
   A-vs-C deltas, and the "Honest limitations" section.
6. **Wrap-up.** Skim `adr/README.md` — every significant choice has an ADR;
   none are hand-wavy.

---

## Honest limitations and known gaps

A clear-eyed account, kept here so it stays visible:

- **Citation_count is `null` for every doc in the manifest** (Roadmap §0
  finding #4). FindPapers therefore ranks on `(is_foundational,
  tag_match_count, year)`, not citation count. ADR-0008 documents the
  substitution. Backfilling via Semantic Scholar is a one-day follow-up.
- **The extractive generator produces blunt, quote-like prose.** It cannot
  fabricate, which is the point in CI, but it isn't the learner-facing
  quality bar. Set `RN_GENERATION__BACKEND=openai` and point at a local
  Ollama for fluent synthesis — that is what M4's `--llm-judge` evaluates.
- **The router is regex-based** and will miss queries that need genuine
  language understanding ("the paper that introduced the technique behind
  ChatGPT"). The default-to-`concept_explanation` fallback + the M2 refusal
  gate make this fail safely; an LLM router can be slotted in behind the
  same `classify_query` signature later.
- **`doc_id` filtering uses Qdrant's filter primitives without a payload
  index.** Fine at our scale (50 docs, ~2k chunks); add a payload index in
  `qdrant_store.py` for production scale.
- **Heuristic faithfulness can only check coverage and marker validity** —
  not whether the cited source actually supports the claim. Use the LLM
  judge for that. The report header always names which judge produced the
  numbers (`Judge: heuristic | llm`).
- **Approximate tokens use 4-chars-per-token.** Honest cross-config
  comparison; not accurate enough for a billing estimate.
- **Golden-set labels are author-annotated**, not exhaustively verified. A
  missing relevant doc at labelling time lowers measured recall artificially.
- **The corpus is sealed and non-redistributable.** Reviewers must run the
  one-time `complete_corpus.py` script. The codebase treats the corpus as a
  variable (paths in config, manifest validated on load), so the same code
  works against a different sealed corpus when the client swaps it in.
- **Bonus tracks B1–B4 are not started.** M0–M4 are complete; the bonus
  query/ingestion UIs, reranker, and vernacular handling remain open.

If something doesn't work for you, run with `RN_LOG_LEVEL=DEBUG
RN_JSON_LOGS=false` and check the structured logs on stderr. Every degraded
component logs the why; nothing is swallowed.

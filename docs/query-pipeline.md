# Query pipeline (M2)

The query pipeline turns a learner's question into a **grounded, citation-carrying
answer — or a graceful refusal**. It is orchestrated by
`research_navigator.generate.pipeline.QueryEngine`, which composes the `retrieve/`
and `generate/` packages over the Qdrant collection that M1 populated.

```
question
   |
   v
[understand]  free text -> inferable metadata filters     retrieve/query.py
   |            - year bounds ("recent" -> year >= floor; "since/before/in YYYY")
   |            - tags from the corpus vocabulary (+ NL aliases)
   |            - content_type / foundational hints
   v
[filter]      QueryAnalysis -> Qdrant Filter               retrieve/filters.py
   |            - Range(year), MatchAny(content_type/tags), MatchValue(foundational)
   v
[retrieve]    hybrid dense+sparse, RRF-fused, filtered      retrieve/hybrid.py
   |            server-side; + a dense-only pass for the
   |            confidence signal (adr-0007)
   |            -- relax tags once if a filter empties the result
   v
[gate]        refuse if confidence < threshold or empty     generate/pipeline.py
   |                                                          (adr-0007)
   v
[cite]        dedup chunks -> numbered citation block        generate/citations.py
   |            (title, authors et al., year, source, section, URL)
   v
[generate]    extractive (default) | OpenAI-compatible       generate/generator.py
   |            every factual sentence carries a [n] marker   (adr-0006)
   v
[validate]    strip out-of-range [n]; no attribution ->      generate/pipeline.py
              refuse. Never let an unattributed claim stand.
```

## Modules

- **`retrieve/query.py`** — `analyze_query` does *filter inference*, not intent
  classification (intent is the M3 router's job). Deterministic and dependency-free:
  year bounds from `recent`/`latest`/`since`/`before`/`in YYYY`, tags matched against
  the corpus vocabulary plus a small natural-language alias map (e.g. "chain of
  thought" → `chain_of_thought`), and content-type/foundational hints. The known-tag
  vocabulary is **passed in** (derived from the manifest) so the corpus stays a
  variable, not a constant — `DEFAULT_TAG_VOCABULARY` is only the fallback.
- **`retrieve/filters.py`** — `build_filter` maps a `QueryAnalysis` to a Qdrant
  `Filter`: `Range` for years, `MatchAny` for content types and tags (OR semantics,
  for recall), `MatchValue` for `is_foundational`. `include_tags=False` produces the
  relaxed filter used by the fallback. All filtering is expressed in Qdrant
  primitives so it runs **server-side**, never as a Python post-filter.
- **`retrieve/hybrid.py`** — `HybridRetriever.retrieve` issues one fused
  `query_points` with two prefetch branches (dense cosine + sparse BM25, each
  carrying the filter) combined by RRF (ADR-0004), then a second dense-only query for
  a similarity-calibrated **confidence** number (ADR-0007). Hits come back as typed
  `RetrievedChunk` models; no raw Qdrant payload escapes the module
  (`RetrievedChunk.from_point` isolates the untyped boundary).
- **`generate/citations.py`** — `build_citations` deduplicates by `doc_id`, keeping
  the best-scoring chunk per document and numbering by descending score (so `[1]` is
  the most relevant source). `source_label` renders the brief's source styles
  (`arXiv:<id>`, `Lil'Log`, `Hugging Face Learn`, lab-blog labels); `format_authors`
  applies first-author *et al.* for ≥ 3 authors.
- **`generate/generator.py`** — the `Generator` protocol with two backends
  (ADR-0006): `ExtractiveGenerator` (offline, deterministic, cannot fabricate) and
  `OpenAICompatibleGenerator` (any Ollama/vLLM/OpenAI chat endpoint). A failed LLM
  call raises `GenerationError` rather than returning an empty answer.
- **`generate/prompt.py`** — the fixed `REFUSAL_TEXT`, the grounding `SYSTEM_PROMPT`,
  and `build_user_prompt` (numbered sources + the question).
- **`generate/pipeline.py`** — `QueryEngine` ties it together and owns the M2
  contract; `Answer` carries `status` (`ANSWERED`/`REFUSED`/`ERROR`), text,
  citations, confidence, and the analysis, and `.render()` appends the sources block.
- **`cli/query.py`** — a thin front end: `python -m research_navigator.cli.query
  "question" [--json]`. No business logic; non-zero exit only on `ERROR` (a refusal
  is a valid outcome, not a failure).

## The citation contract (M2 acceptance)

The acceptance criterion is structural, so the pipeline enforces it structurally
rather than trusting the generator:

1. **Refuse on low confidence.** If the top dense cosine similarity is below
   `RN_RETRIEVAL__REFUSAL_MIN_SCORE` (default `0.30`), or retrieval is empty, return
   the fixed refusal message (ADR-0007).
2. **Cite only real, retrieved sources.** Citations are built *from the retrieved
   chunks*, deduplicated and numbered; the generator is handed exactly those numbered
   sources.
3. **Validate every marker.** After generation, `[n]` markers are checked against the
   real citation set — out-of-range markers (a model inventing `[9]`) are stripped.
4. **No unattributed answer.** If validation leaves an answer with no valid `[n]`, it
   is converted to a refusal. An unattributed claim is never allowed to stand.

So a fabricated citation cannot survive, and a hallucinated answer degrades to a
refusal — independent of which generation backend is configured.

## Configuration

All M2 knobs live in `research_navigator.config.Settings` (env prefix `RN_`, nested
delimiter `__`); nothing in `src/` hardcodes a model, URL, or threshold.

- **`RN_RETRIEVAL__*`** — `top_k` (5, surfaced sources), `candidate_k` (20, fused
  chunks before dedup), `prefetch_limit` (40, per-branch candidates before fusion),
  `refusal_min_score` (0.30), `recent_year_floor` (2024).
- **`RN_GENERATION__*`** — `backend` (`extractive` | `openai`), `max_sentences` (6),
  and the OpenAI-compatible client settings `llm_base_url` (default local Ollama),
  `llm_model`, `llm_api_key`, `llm_temperature` (0.0), `llm_max_tokens`, `llm_timeout`.

## Running a query

```bash
# Offline / deterministic default (no key, no LLM server): extractive backend.
python -m research_navigator.cli.query "What is retrieval-augmented generation?"

# JSON output (status, citations, confidence) for scripting / eval.
python -m research_navigator.cli.query "Compare DPO and KTO" --json

# Fluent synthesis via a local OSS model (OSS-first, no secret):
#   ollama serve && ollama pull llama3.1:8b-instruct-q4_K_M
export RN_GENERATION__BACKEND=openai
python -m research_navigator.cli.query "How does chain-of-thought prompting work?"
```

Tune the refusal floor against your embedding model without touching code, e.g.
`export RN_RETRIEVAL__REFUSAL_MIN_SCORE=0.35`. The default is calibrated for the
production `bge-small` dense model (ADR-0001); M4 will re-tune it on the held-out
out-of-corpus set.

## Testing without a server or network

The same hermetic setup as ingestion: tests point `RN_QDRANT_URL` at `:memory:` and
set `RN_USE_OFFLINE_EMBEDDER=1`. Query-understanding, filter-building, and citation
rendering are pure unit tests; the end-to-end pipeline tests upsert a tiny fixture
corpus into in-process Qdrant and assert the contract — valid citations trace to
retrieved chunks, low confidence and empty filters refuse, filters are applied inside
Qdrant, out-of-range markers are stripped, and an unattributed answer becomes a
refusal.

> The offline hashing embedder has no semantic geometry, so the refusal tests force
> the real code paths deterministically (an impossibly high threshold; a filter that
> matches no document) rather than relying on embedding quality — see ADR-0007.

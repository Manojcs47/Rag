# ADR-0006 — Extractive-by-default generation behind an OpenAI-compatible backend

- **Status:** Accepted
- **Date:** 2025 (M2)
- **Deciders:** project owner (intern) + reviewer
- **Relates to:** M2 (query pipeline), ADR-0001 (local-first models), ADR-0007 (refusal)

## Context

M2 must turn retrieved chunks into an answer in which **every factual sentence
carries an inline `[n]` citation**, or refuse. This is the question the roadmap left
open ("Decisions to lock" #2: generation LLM — local via Ollama vs API). Two forces
pull against each other:

- The grading environment and CI must run **without secrets, without paid quota, and
  ideally offline** — the same constraint that shaped ADR-0001 for embeddings. A
  pipeline whose answer step *requires* a hosted LLM cannot be tested hermetically.
- A real learner-facing answer wants an actual LLM doing multi-source synthesis, and
  the brief is **OSS-first**: prefer self-hosted (Ollama, vLLM) unless a proprietary
  option is materially better.

We also need the citation contract to be enforceable regardless of backend. An LLM
can be *prompted* to cite, but prompting alone cannot *guarantee* no unattributed or
fabricated citation — the structural enforcement lives in the pipeline (ADR-0007),
not in the model.

## Decision

Generation sits behind a small **`Generator` protocol** (`name`, `generate(query,
sources) -> GeneratedAnswer`), mirroring the `DenseEmbedder`/`SparseEmbedder`
pattern from ingestion, with two implementations selected by config
(`RN_GENERATION__BACKEND`):

- **`ExtractiveGenerator` (default).** Offline, deterministic, zero dependencies
  beyond the standard library. For each retrieved source it picks the sentence with
  the highest query-term overlap and attaches that source's `[n]` marker. It **cannot
  fabricate** a claim or a citation — every sentence it emits is lifted from a real
  retrieved chunk and tagged with that chunk's number. This is what CI, tests, and a
  keyless `docker compose up` demo run.
- **`OpenAICompatibleGenerator`.** Lazily POSTs to any OpenAI-compatible
  `/chat/completions` endpoint (`RN_GENERATION__LLM_BASE_URL`, default local Ollama
  `http://localhost:11434/v1`). The grounding rules live in a system prompt
  (cite every factual sentence, use only the supplied source numbers, paraphrase
  rather than quote, refuse if the sources are insufficient). It does real
  multi-source synthesis when a learner wants prose.

Everything is configurable — backend, model id, base URL, API key (`SecretStr`),
temperature (default `0.0` for reproducibility, M5), max tokens, timeout — so moving
from the offline default to local Ollama to a hosted API is a config change, not a
code change.

Crucially, **the citation guarantee does not depend on which backend runs.** The
pipeline validates the markers a generator emits against the real, deduplicated
citation set: out-of-range markers are stripped, and an answer left with no valid
attribution is converted to a refusal (ADR-0007). A hallucinating LLM degrades to a
refusal; it cannot produce a fabricated citation.

## Consequences

- The whole M2 pipeline is testable offline and deterministically: the extractive
  backend exercises retrieval → dedup → citation rendering → marker validation with
  no network and no key.
- A reviewer gets a working demo immediately (`backend=extractive`), then can opt
  into fluent synthesis by starting Ollama and setting `backend=openai` — no secret
  required for the OSS path.
- The extractive default produces blunt, quote-like prose, not flowing synthesis;
  it is a correctness-and-grounding baseline, not the final learner experience. The
  `openai` backend is the quality path and is what M3/M4 will evaluate.
- `httpx` is the only added dependency, and it is imported lazily inside the LLM
  backend so the offline path never pays for it.
- A failed or timed-out LLM call raises `GenerationError`, which the pipeline turns
  into an explicit `ERROR` status — never a silent empty answer (M5, "no silent
  failure").

## Alternatives considered

- **Require a hosted LLM (OpenAI/Anthropic) for generation** — best fluency, but
  re-introduces the secret/cost/network-in-CI problems ADR-0001 spent effort
  avoiding, and conflicts with OSS-first. Rejected as the *default*; still reachable
  by pointing the same backend at `api.openai.com`.
- **LangChain/LlamaIndex generation chains** — heavier abstractions than a single
  typed `generate` call needs, and they obscure exactly the marker-validation step
  the brief grades. Rejected; LangGraph enters in M3 for *orchestration*, not for
  wrapping the LLM call.
- **Trust the LLM's prompted citations without validation** — cannot satisfy the M2
  acceptance criterion ("no citation is fabricated"). Rejected; markers are always
  validated structurally regardless of backend.
- **Templated/extractive only, no LLM path at all** — fully safe but never produces
  real synthesis, which M3's `ConceptExplanation`/`CompareApproaches` routes need.
  Rejected in favour of keeping both behind one protocol.

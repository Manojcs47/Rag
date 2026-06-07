# ADR-0008 — Agentic routing with LangGraph: deterministic router, typed tools, metadata-only FindPapers

- **Status:** Accepted
- **Date:** 2025 (M3)
- **Deciders:** project owner (intern) + reviewer
- **Relates to:** M3 (agent layer), ADR-0006 (generation), ADR-0007 (refusal)

## Context

M3 asks for a LangGraph state machine that routes a learner's query into one of
six sub-capabilities (concept_explanation, paper_deep_dive, compare_approaches,
recent_developments, find_papers, out_of_scope) with an explicit, serialisable
state and at least one tool call inside a node. Three sub-decisions were open:

1. **How does the router classify?** An LLM-based classifier reads naturally
   but reintroduces the secret/cost/network-in-CI problem ADR-0001 / ADR-0006
   set up the offline path to avoid, and makes routing non-deterministic.
2. **What does "tool call" mean?** LangChain `@tool` decorators with an LLM
   driving them, or explicit typed helper functions invoked from nodes.
3. **How does FindPapers rank?** The brief lists `is_foundational`,
   `citation_count`, `year` — but `citation_count` is null for every document
   in the sealed corpus (Roadmap §0 finding #4), so the ranking signal needs a
   concrete fallback.

## Decision

- **Heuristic, regex-based router.** A small ordered set of patterns matches
  comparison ("compare X and Y", "X vs Y"), reading-list intent ("recommend
  papers"), explicit paper references (arXiv id, distinctive title words,
  method aliases), recency markers, and a small off-topic vocabulary. The
  ordering is precedence. Ambiguous queries fall through to
  `concept_explanation`; the M2 refusal gate (ADR-0007) catches genuinely
  off-topic content that slips past the off-topic regex. The classifier is a
  pure function — `classify_query(query, documents) -> (route, hints)` —
  trivially unit-testable per route.

- **Tools are typed Python helpers, recorded in state.** Three helpers live in
  `agents/tools.py`: `corpus_metadata_lookup`, `date_math`,
  `rank_papers_for_reading_list`. Nodes invoke them directly and append a
  `ToolInvocation` record to `AgentState.tool_log`, so every step is
  inspectable in JSON output. This satisfies the brief literally ("at least
  one tool call within an agent node, e.g. a structured corpus-metadata
  lookup, or a date-math helper for RecentDevelopments") without dragging in
  an LLM tool-calling loop that would be untestable offline.

- **FindPapers ranks on `(is_foundational, tag_match_count, year)`.**
  Citation-count is null across the corpus, so using it would be either a
  silent zero or a dishonest "we use citation_count" claim that does nothing.
  We document the substitution here, expose `foundational_first` as a routing
  hint (set when the query says "seminal/foundational/classic"), and fall back
  to a metadata-only ranking. FindPapers therefore returns a templated reading
  list — no LLM generation — which matches the brief's instruction to "rely on
  metadata filters... rather than free-form generation".

- **State is a pydantic `AgentState`.** Explicit fields for query, route,
  hints, answer, tool_log. JSON-serialisable, validates on construction,
  round-trips via `model_dump`. LangGraph 1.x accepts pydantic state directly.

- **One small additive refactor to `QueryEngine`.** The existing `answer()`
  flow is preserved; we expose three new public hooks (`answer_with_filter`,
  `answer_with_analysis`, `answer_from_chunks`) and a `retrieve_with_filter`
  passthrough so M3 nodes can plug custom retrieval into M2's
  cite-generate-validate machinery without duplicating it.

- **`doc_id` filtering uses Qdrant filter primitives, no payload index.**
  PaperDeepDive and CompareApproaches build `models.Filter` directly on the
  `doc_id` payload field. At our corpus scale (50 docs, ~2k chunks), unindexed
  filtering is acceptable; adding the index later is a one-line change in M1.

## Consequences

- The router is reproducible — same query, same route, no API call. CI runs
  the full agent layer offline.
- Each node's tool calls are visible in `AgentState.tool_log`, which the CLI
  surfaces in both human and JSON modes (transparency, per the brief's bonus
  track B1 spirit even without a UI).
- FindPapers cannot fabricate a recommendation — every entry is a real
  manifest document with a real URL. If no document matches the topic, the
  node refuses.
- The router will miss cases that need real language understanding ("which
  paper introduced the technique behind ChatGPT?"). The default-to-concept
  fallback + M2 refusal makes this fail safely; an optional LLM router can be
  added later behind the same `classify_query` interface.

## Alternatives considered

- **LLM-driven router.** Better generalisation, but reintroduces secret +
  network dependence in CI, and makes routing non-deterministic — bad for M4
  evaluation reproducibility. Rejected as the default. The `classify_query`
  signature leaves room for a future LLM-backed implementation.
- **LangChain `@tool` + ToolNode loop.** Idiomatic for ReAct-style agents,
  but unnecessary here: each route has a fixed, small set of tool calls
  that map 1:1 to typed Python functions. The `@tool` decorator would add
  abstraction without buying anything.
- **Use `citation_count` despite all values being null.** Equivalent to
  ignoring it. Better to be explicit about the substitution.
- **TypedDict state with channel reducers.** Works, but pydantic gives us
  validation + clearer types and matches the rest of the codebase style. No
  per-field reducers needed because each run touches at most two nodes
  (router + one sub-capability).

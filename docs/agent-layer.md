# Agent layer (M3)

The agent layer routes each learner question into one of six specialised
sub-capabilities, then runs that single node and returns. Orchestration uses
LangGraph; classification, tools, and node bodies are plain typed Python.
```
question
|
v
[router]   regex-based classification + hint extraction       agents/router.py
|          - compare_approaches: "vs", "compare X and Y"
|          - find_papers:        "reading list", "recommend"
|          - paper_deep_dive:    arXiv id, title word, method alias
|          - recent_developments:"recent", "latest", "SOTA"
|          - out_of_scope:       off-topic regex + no in-scope signal
|          - concept_explanation:default
v
[conditional edge by route]
|
v  (one of the six sub-capability nodes)
[concept_explanation]   QueryEngine.answer(q)
[paper_deep_dive]       Filter(doc_id=X) -> QueryEngine.answer_with_filter
[compare_approaches]    per-target retrieve -> answer_from_chunks(union)
[recent_developments]   date_math() -> forced year_min -> answer_with_analysis
[find_papers]           metadata-only ranking, no LLM
[out_of_scope]          polite decline
|
v
END (state.answer populated; state.tool_log records any tool calls)
```
## State

```python
class AgentState(BaseModel):
    query: str
    route: AgentRoute | None
    hints: RoutingHints
    answer: Answer | None
    tool_log: list[ToolInvocation]
```

Pydantic-typed and JSON-serialisable. `hints` carries `paper_doc_id`,
`compare_targets`, `foundational_only`, `recent_only`. `tool_log` records every
tool call a node made (name, args, result summary) so a run is inspectable.

## Tools

- **`corpus_metadata_lookup`** — filters the manifest by tags / content_type /
  is_foundational / year / doc_id. Used by PaperDeepDive (existence check),
  CompareApproaches (resolve target -> doc_ids), FindPapers (candidate set).
- **`date_math`** — computes a year-floor `months_back` months before today.
  Used by RecentDevelopments to derive the recency cutoff.
- **`rank_papers_for_reading_list`** — ranks candidates on
  `(is_foundational, tag_match_count, year)`. Used by FindPapers. See ADR-0008
  for why `citation_count` is not in the signal (null across the corpus).

## Visualising the graph

```bash
uv run python -m research_navigator.cli.agent --visualize docs/agent-graph.md
```

This writes a Mermaid diagram GitHub renders natively. Embed it in
`ARCHITECTURE.md` or open `docs/agent-graph.md` directly.

## Running

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
```

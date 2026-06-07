"""CLI for the M3 agent layer.

    python -m research_navigator.cli.agent "compare DPO and KTO"
    python -m research_navigator.cli.agent "recommend papers on RAG" --json
    python -m research_navigator.cli.agent --visualize docs/agent-graph.md

Thin by design: parse args, build the graph, run one query, render.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research_navigator.agents.graph import build_agent_graph, save_graph_visualization
from research_navigator.agents.state import AgentState
from research_navigator.config import get_settings
from research_navigator.logging import configure_logging


def _result_dict(state: AgentState) -> dict[str, object]:
    return {
        "query": state.query,
        "route": state.route.value if state.route else None,
        "hints": state.hints.model_dump(),
        "tool_log": [t.model_dump() for t in state.tool_log],
        "answer": state.answer.model_dump() if state.answer else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research_navigator.cli.agent")
    parser.add_argument("query", nargs="?", help="The question to route + answer.")
    parser.add_argument("--json", action="store_true", help="Emit full JSON output.")
    parser.add_argument(
        "--visualize",
        metavar="PATH",
        help="Write a Mermaid diagram of the graph to PATH and exit.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    if args.visualize:
        out = save_graph_visualization(Path(args.visualize))
        print(f"Wrote graph to {out}")
        return 0

    if not args.query:
        parser.error("query is required (or pass --visualize PATH)")

    graph = build_agent_graph(settings)
    final_dict = graph.invoke(AgentState(query=args.query))
    final = AgentState.model_validate(final_dict)

    if args.json:
        print(json.dumps(_result_dict(final), indent=2, ensure_ascii=False, default=str))
    else:
        print(f"Route: {final.route.value if final.route else '?'}")
        if final.tool_log:
            print(f"Tool calls: {', '.join(t.name for t in final.tool_log)}")
        print()
        if final.answer:
            print(final.answer.render())

    if final.answer and final.answer.status.value == "error":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

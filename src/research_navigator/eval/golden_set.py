"""Load and validate the golden-set JSONL (M4).

Each line is one labelled question. The schema is intentionally small and
opinionated — every field is required so a typo in the data file is a loud
:class:`pydantic.ValidationError`, never a silent skip downstream.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from research_navigator.agents.state import AgentRoute


class GoldenItem(BaseModel):
    """One labelled query in the golden set."""

    id: str
    query: str
    route: AgentRoute
    expected_doc_ids: list[str] = Field(default_factory=list)
    """Doc_ids that *should* appear in the answer's citations (retrieval truth)."""

    expected_sections: list[str] = Field(default_factory=list)
    """Optional section names (informational; not currently scored)."""

    is_out_of_corpus: bool = False
    """If True, the system should refuse rather than answer."""

    notes: str = ""

    @property
    def expected_refusal(self) -> bool:
        return self.is_out_of_corpus or self.route is AgentRoute.OUT_OF_SCOPE


def load_golden_set(path: Path) -> list[GoldenItem]:
    """Read ``path`` as JSONL and validate every line.

    Raises:
        FileNotFoundError: if ``path`` doesn't exist.
        pydantic.ValidationError: on the first malformed line (loud, not silent).
    """
    if not path.is_file():
        raise FileNotFoundError(f"golden set not found: {path}")
    items: list[GoldenItem] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw or raw.startswith("//"):
            continue
        try:
            items.append(GoldenItem.model_validate_json(raw))
        except Exception as exc:  # re-raise with line context
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return items

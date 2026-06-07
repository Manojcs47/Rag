# ADR-0002 — uv for packaging and environment management

- **Status:** Accepted
- **Date:** 2025 (M0, ratified M1)
- **Deciders:** project owner (intern) + reviewer

## Context

The project needs one tool to manage the Python version, the virtual environment,
the dependency graph, and a reproducible lockfile. The candidates were `pip` +
`venv` + `pip-tools`, `poetry`, `pdm`, and `uv`. The brief values reproducibility
and a low-friction setup that a reviewer can run in one or two commands.

## Decision

Use **uv** as the single packaging/environment tool.

- `pyproject.toml` declares runtime deps and a `dev` dependency group; the
  `uv_build` backend builds the package.
- `uv.lock` is committed and pins the full resolved graph for reproducible installs.
- `requires-python = ">=3.13"` and `.python-version` pin the interpreter; uv will
  fetch it if absent.
- Standard commands: `uv sync` (install), `uv run <cmd>` (run in the env),
  `uv add/remove` (manage deps). The `Makefile` wraps these so contributors don't
  need to memorise them.

## Consequences

- One fast tool replaces three; resolution and installs are markedly quicker than
  pip/poetry, which matters as `torch`/`sentence-transformers` land in M1.
- The committed lockfile makes "works on my machine" reproducible across the
  reviewer's machine and CI.
- Contributors must have `uv` installed (one `curl` / `pipx install uv`); this is
  the only bootstrap requirement.
- uv is younger than pip/poetry; the lockfile format is uv-specific. Mitigated by
  uv's stability and the fact that `pyproject.toml` stays tool-agnostic, so a future
  migration would not require code changes.

## Alternatives considered

- **pip + venv + pip-tools** — ubiquitous but manual; three tools, slower, easy to
  drift from the lockfile. Rejected for friction.
- **poetry** — mature and popular, but slower resolution and a heavier workflow; its
  build/PEP-621 story has historically lagged. Rejected on speed/ergonomics.
- **pdm** — close to uv in spirit; rejected only because uv's speed and momentum
  made it the stronger default.

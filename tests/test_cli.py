"""Tests for the ingest CLI (thin entrypoint): subcommands, exit codes, JSON output."""

from __future__ import annotations

import json

import pytest

from research_navigator.cli import ingest as cli
from research_navigator.config import Settings

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    """Point the CLI at the fixture settings instead of the real environment."""
    monkeypatch.setattr(cli, "get_settings", lambda: settings)


def test_validate_command_exit_zero(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["validate"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "documents" in out
    assert out["errors"] == []


def test_ingest_command_outputs_report(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["ingest"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["total_added"] > 0
    assert "total_writes" in out


def test_ingest_single_doc_flag(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["ingest", "--doc", "hf-nlp-ch09"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["documents"]) == 1


def test_reindex_command(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["reindex"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["total_added"] > 0


def test_stats_command(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["stats"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert "total_chunks" in out
    assert "by_content_type" in out


def test_missing_subcommand_errors() -> None:
    with pytest.raises(SystemExit):
        cli.main([])

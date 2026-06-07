"""Smoke tests proving the package imports and core wiring works."""

from __future__ import annotations

from research_navigator import __version__
from research_navigator.config import Settings, get_settings
from research_navigator.logging import configure_logging, get_logger


def test_version_is_string() -> None:
    assert isinstance(__version__, str)
    assert __version__


def test_settings_have_sane_defaults() -> None:
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.qdrant_collection
    assert settings.qdrant_url.startswith("http")


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_logging_configures_and_logs() -> None:
    configure_logging(level="INFO", json_logs=False)
    log = get_logger("test")
    log.info("smoke", value=1)

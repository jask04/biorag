"""Smoke tests for project scaffolding and settings."""

import pytest

from biorag import __version__
from biorag.config import Settings


def test_version_is_set() -> None:
    assert __version__ == "0.1.0"


def test_settings_optional_until_accounts_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in ("GOOGLE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings()
    assert settings.google_api_key is None
    assert settings.qdrant_url is None
    assert settings.qdrant_api_key is None

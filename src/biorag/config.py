"""Runtime configuration loaded from environment / .env.

Secrets (the Gemini key, the Qdrant key) live only in ``.env`` locally and
in the deploy secret store in production. They are read here and never
logged. ``QDRANT_*`` becomes required on Day 4 and ``GOOGLE_API_KEY`` on
Day 9; until then they stay optional so the project runs with no accounts.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings for biorag."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    qdrant_url: str | None = Field(default=None, alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings, parsed once."""
    return Settings()

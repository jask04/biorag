"""Keep the public demo infrastructure warm.

The script intentionally uses only the Python standard library so scheduled
jobs can run quickly without installing the full RAG stack. It performs:

1. A read-only Qdrant Cloud request against ``GET /collections``.
2. A public HTTP request to the Hugging Face Space app URL.

Both requests are enough to prove the services are reachable and to register
activity without mutating any application data.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_HF_SPACE_URL = "https://jask04-biorag.hf.space"
DEFAULT_RETRIES = 6
DEFAULT_RETRY_DELAY_SECONDS = 30
DEFAULT_TIMEOUT_SECONDS = 30


class KeepaliveError(RuntimeError):
    """Raised when a keepalive target cannot be reached."""


def load_local_env(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE pairs from .env without overriding real env vars."""
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment."""
    raw = os.environ.get(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise KeepaliveError(f"{name} must be an integer") from exc
    if value <= 0:
        raise KeepaliveError(f"{name} must be positive")
    return value


def open_json(url: str, headers: dict[str, str], timeout: int) -> dict[str, Any]:
    """Fetch JSON from a URL and return the decoded object."""
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise KeepaliveError(f"{url} returned HTTP {exc.code}: {body[:200]}") from exc
    except urllib.error.URLError as exc:
        raise KeepaliveError(f"{url} request failed: {exc.reason}") from exc

    if not 200 <= status < 300:
        raise KeepaliveError(f"{url} returned HTTP {status}")

    try:
        decoded = json.loads(body)
    except json.JSONDecodeError as exc:
        raise KeepaliveError(f"{url} did not return valid JSON") from exc
    if not isinstance(decoded, dict):
        raise KeepaliveError(f"{url} returned unexpected JSON")
    return decoded


def request_with_retries(
    label: str,
    request_fn: Any,
    retries: int,
    retry_delay_seconds: int,
) -> None:
    """Run a request, retrying transient failures."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request_fn()
            print(f"{label}: ok")
            return
        except KeepaliveError as exc:
            last_error = exc
            if attempt == retries:
                break
            print(f"{label}: attempt {attempt} failed; retrying")
            time.sleep(retry_delay_seconds)

    raise KeepaliveError(f"{label}: failed after {retries} attempts: {last_error}")


def ping_qdrant(timeout: int) -> None:
    """Make a read-only Qdrant collections request."""
    qdrant_url = os.environ.get("QDRANT_URL", "").rstrip("/")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")
    if not qdrant_url or not qdrant_api_key:
        raise KeepaliveError("QDRANT_URL and QDRANT_API_KEY are required")

    response = open_json(
        f"{qdrant_url}/collections",
        headers={
            "api-key": qdrant_api_key,
            "User-Agent": "biorag-keepalive/1.0",
        },
        timeout=timeout,
    )
    collections = response.get("result", {}).get("collections")
    if not isinstance(collections, list):
        raise KeepaliveError("Qdrant response did not include collections")

    print(f"Qdrant collections visible: {len(collections)}")


def ping_hf_space(timeout: int) -> None:
    """Request the public Hugging Face Space app URL."""
    hf_space_url = os.environ.get("HF_SPACE_URL", DEFAULT_HF_SPACE_URL).rstrip("/")
    request = urllib.request.Request(
        hf_space_url,
        headers={"User-Agent": "biorag-keepalive/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        raise KeepaliveError(f"{hf_space_url} returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise KeepaliveError(f"{hf_space_url} request failed: {exc.reason}") from exc

    if not 200 <= status < 400:
        raise KeepaliveError(f"{hf_space_url} returned HTTP {status}")


def main() -> None:
    """Run all keepalive checks."""
    load_local_env()
    timeout = env_int("KEEPALIVE_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)
    retries = env_int("KEEPALIVE_RETRIES", DEFAULT_RETRIES)
    retry_delay_seconds = env_int(
        "KEEPALIVE_RETRY_DELAY_SECONDS",
        DEFAULT_RETRY_DELAY_SECONDS,
    )

    request_with_retries(
        "qdrant",
        lambda: ping_qdrant(timeout),
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
    )
    request_with_retries(
        "hugging-face-space",
        lambda: ping_hf_space(timeout),
        retries=retries,
        retry_delay_seconds=retry_delay_seconds,
    )


if __name__ == "__main__":
    main()

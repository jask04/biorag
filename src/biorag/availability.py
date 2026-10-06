"""Read-only demo collection check, usable with the runner's standard library."""

from __future__ import annotations

import json
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEMO_COLLECTION = "biorag_BAAI__bge_small_en_v1_5"


def validate_collection(result: dict[str, Any]) -> int:
    """Reject an empty, unhealthy, or incompatible demo index."""
    if result.get("status") != "green":
        raise ValueError("The demo collection is not healthy")
    points = result.get("points_count")
    if not isinstance(points, int) or points <= 0:
        raise ValueError("The demo collection has no indexed passages")
    vectors = result.get("config", {}).get("params", {}).get("vectors", {})
    if vectors.get("size") != 384 or vectors.get("distance") != "Cosine":
        raise ValueError("The demo collection does not match the BGE-small embedder")
    return points


def main() -> int:
    """Check the existing collection without creating keys or changing data."""
    url = os.environ.get("QDRANT_URL", "").rstrip("/")
    key = os.environ.get("QDRANT_API_KEY", "")
    if not url or not key:
        print("QDRANT_URL and QDRANT_API_KEY must be configured", file=sys.stderr)
        return 1
    request = Request(
        f"{url}/collections/{DEMO_COLLECTION}", headers={"api-key": key}
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
        points = validate_collection(payload.get("result", {}))
    except HTTPError as exc:
        print(f"Qdrant returned HTTP {exc.code}", file=sys.stderr)
        return 1
    except (URLError, TimeoutError, OSError):
        print("Qdrant connection failed", file=sys.stderr)
        return 1
    except (ValueError, AttributeError, TypeError):
        print(
            "Qdrant demo collection is missing, empty, or incompatible", file=sys.stderr
        )
        return 1
    print(f"Qdrant demo collection is healthy ({points} indexed passages).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The availability check must not pass merely because the server is reachable."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from biorag.availability import validate_collection

HEALTHY: dict[str, Any] = {
    "status": "green",
    "points_count": 5082,
    "config": {"params": {"vectors": {"size": 384, "distance": "Cosine"}}},
}


def test_healthy_demo_collection() -> None:
    assert validate_collection(HEALTHY) == 5082


@pytest.mark.parametrize("points", [0, None, "5082"])
def test_empty_or_invalid_point_count(points: object) -> None:
    result = deepcopy(HEALTHY)
    result["points_count"] = points
    with pytest.raises(ValueError, match="no indexed passages"):
        validate_collection(result)


def test_wrong_embedding_dimension() -> None:
    result = deepcopy(HEALTHY)
    result["config"]["params"]["vectors"]["size"] = 768
    with pytest.raises(ValueError, match="embedder"):
        validate_collection(result)


def test_unhealthy_collection() -> None:
    result = deepcopy(HEALTHY)
    result["status"] = "red"
    with pytest.raises(ValueError, match="not healthy"):
        validate_collection(result)

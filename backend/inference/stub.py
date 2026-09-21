"""Deterministic stand-in for the Phase 1 checkpoint.

It is not a model and never pretends to be one: `model_version` says `stub`
so nothing downstream — a stored result, an MLflow comparison, a screenshot —
can be mistaken for a real prediction.

Its probability is the file's content hash folded into [0, 1). That makes it
deterministic per image (the same upload always yields the same result, so the
UI and tests are stable) while still spreading across the decision threshold
and the abstention band, which a fixed constant would not.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


class StubPredictor:
    model_version = "stub:v0"

    def __init__(self, forced_probability: float | None = None) -> None:
        # Set by tests, and by `settings.stub_probability` in dev, to drive the
        # product into a chosen FR-04 state — including abstention — without
        # hunting for a file whose hash happens to land there.
        self._forced = forced_probability

    def probability(self, image_path: Path) -> float:
        if self._forced is not None:
            return self._forced
        digest = hashlib.sha256(image_path.read_bytes()).digest()
        return int.from_bytes(digest[:4], "big") / 2**32

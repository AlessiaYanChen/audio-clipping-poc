from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class BoundaryCandidate:
    time_s: float
    score: float
    reason: str
    quote: str | None = None
    boundary_type: str | None = None
    confidence: float | None = None


def propose_candidates() -> List[BoundaryCandidate]:
    return []

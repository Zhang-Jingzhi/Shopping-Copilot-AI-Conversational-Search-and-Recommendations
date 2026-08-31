from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


IntentType = Literal["buying", "browsing"] | None


@dataclass(frozen=True)
class IntentResult:
    """Stable hand-off contract for retrieval, state, and ranking modules."""

    raw_query: str
    normalized_query: str
    intent_type: IntentType
    intent_confidence: float
    slots: dict[str, list[str] | float]
    hard_constraints: dict[str, list[str] | float]
    filter_constraints: dict[str, list[str] | float]
    soft_preferences: dict[str, list[str] | float]
    keyword_query: str
    semantic_query: str
    ambiguity_flags: list[str] = field(default_factory=list)
    override_detected: bool = False
    decision_evidence: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

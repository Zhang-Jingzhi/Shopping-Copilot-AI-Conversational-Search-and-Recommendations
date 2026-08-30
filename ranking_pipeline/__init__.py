"""Ranking and clarification policy package for the D component."""

from __future__ import annotations

import sys
from pathlib import Path


# Make the package runnable from either the repository root or from
# retrieval-and-reranking. This package depends on the sibling techjam_agent
# package, while techjam_agent exposes the repository root in its own import
# bootstrap.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_TECHJAM_ROOT = _REPOSITORY_ROOT / "retrieval-and-reranking"
if str(_TECHJAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_TECHJAM_ROOT))

from ranking_pipeline.contextual_ranking import (
    CandidatePoolMetrics,
    ConstraintConflict,
    HybridContextualReranker,
    LocalQwenRanker,
    PolicyDecision,
    RecommendationClarificationPolicy,
    assess_candidate_pool,
    choose_clarification_attribute,
    choose_rerank_strategy,
)
from ranking_pipeline.prompt import LLMRankResult, build_rerank_prompt, parse_rerank_output
from ranking_pipeline.qwen_reranker import Qwen3Reranker

__all__ = [
    "ConstraintConflict",
    "CandidatePoolMetrics",
    "HybridContextualReranker",
    "LLMRankResult",
    "LocalQwenRanker",
    "PolicyDecision",
    "Qwen3Reranker",
    "RecommendationClarificationPolicy",
    "assess_candidate_pool",
    "choose_clarification_attribute",
    "choose_rerank_strategy",
    "build_rerank_prompt",
    "parse_rerank_output",
]

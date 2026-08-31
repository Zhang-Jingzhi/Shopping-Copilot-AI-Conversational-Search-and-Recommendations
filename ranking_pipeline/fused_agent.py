"""Standalone fusion agent for state-memory + intent-recognition + ranking.

This module intentionally does NOT modify the existing ``ranking_pipeline``
agent or reranker.  It is an opt-in agent that:

* keeps the official ``OverrideAwareRequirementsCollector`` as the source of
  disclosed dialogue requirements,
* maintains an explicit ``StateMemoryManager`` snapshot per session,
* maintains an explicit ``IntentRouter`` result per turn,
* fuses those three sources into one ranking view, and
* applies a standalone ``FusedContextualReranker`` that consumes the fused
  requirements plus state/intent penalties and boosts.

The default candidate pool is still generated from the official collector so
it matches the locked-exact Top-50 pool unless ``--fuse-retrieval`` is used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from techjam_agent.contracts import (
    Candidate,
    CandidateSet,
    RankedCandidate,
    RerankResult,
    Requirements,
)
from techjam_agent.query import parse_intent, tokenize
from techjam_agent.ranking import LockedWeightedRrfTop10Reranker
from techjam_agent.retrieval import (
    ExactDenseTop50CandidateGenerator,
    LiteTop50CandidateGenerator,
    VISIBLE_FIELDS,
    _text,
)

from ranking_pipeline.context import ShortTermSummary, profile_features
from ranking_pipeline.contextual_ranking import (
    choose_rerank_strategy,
    evaluate_hard_constraints,
    filter_hard_conflicts,
)
from ranking_pipeline.memory_context import (
    intent_to_context,
    intent_to_requirements,
    merge_profile_with_snapshot,
    snapshot_to_requirements,
)
from ranking_pipeline.override_aware_agent import OverrideAwareRequirementsCollector
from ranking_pipeline.qwen_reranker import Qwen3Reranker

from state_memory import StateMemoryManager
from intent_router import IntentRouter, load_catalog_brands, load_catalog_categories


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLEAN_ADAPTER = (
    REPOSITORY_ROOT
    / "ranking_pipeline"
    / "checkpoints"
    / "0.6Blora_aligned_from_shopping_lora_epoch1"
)
FALLBACK_ADAPTER = (
    REPOSITORY_ROOT
    / "ranking_pipeline"
    / "checkpoints"
    / "qwen3-reranker-0.6B-shopping-lora"
)


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, Mapping):
        return [str(item) for item in value.values() if item not in (None, "", [], {})]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _norm(value: object) -> str:
    return " ".join(tokenize(str(value)))


def _unique(values: Iterable[object]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        key = _norm(text)
        if not text or not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return tuple(result)


def _first(*values: object, default: str = "") -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return default


@dataclass(frozen=True)
class FusedRequirements:
    """The ranking view after merging official/state/intent signals."""

    requirements: Requirements
    exclusions: tuple[str, ...]
    override_active: bool
    source: str = "fused"


def collect_exclusions(
    snapshot: Any | None,
    intent_result: Any | None,
) -> tuple[str, ...]:
    """Collect hard negatives from state memory and intent recognition."""

    values: list[str] = []
    must_not = getattr(snapshot, "must_not_match", None) or {}
    if isinstance(must_not, Mapping):
        for value in must_not.values():
            values.extend(_flatten_values(value))
    if intent_result is not None:
        hard = getattr(intent_result, "hard_constraints", None) or {}
        for name, value in hard.items():
            if str(name).endswith("_exclude"):
                values.extend(_flatten_values(value))
    return _unique(values)


def fuse_requirements(
    official: Requirements,
    snapshot: Any | None = None,
    intent_result: Any | None = None,
    *,
    override_turns: Sequence[int] = (),
    include_exclusions: bool = True,
) -> Requirements:
    """Merge official collector, state-memory, and intent requirements.

    The official collector remains authoritative for evaluator-disclosed text.
    State memory and intent recognition add structured slots.  On override
    turns, state-memory history is deliberately suppressed so stale soft
    preferences cannot reintroduce the displaced value.
    """

    state_req = snapshot_to_requirements(snapshot) if snapshot is not None else Requirements("", (), ())
    intent_req = intent_to_requirements(intent_result) if intent_result is not None else Requirements("", (), ())

    debug = getattr(snapshot, "debug", None) or {}
    override_active = (
        bool(override_turns)
        or bool(getattr(intent_result, "override_detected", False))
        or bool(debug.get("category_overridden"))
        or bool(debug.get("intent_changed"))
    )

    category = _first(
        official.category,
        intent_req.category,
        state_req.category,
        default="clothing item",
    )

    hard_values = list(official.hard_constraints)
    hard_values.extend(intent_req.hard_constraints)
    if not override_active:
        hard_values.extend(state_req.hard_constraints)
    hard_constraints = _unique(hard_values)

    soft_values = list(official.soft_preferences)
    soft_values.extend(intent_req.soft_preferences)
    if not override_active:
        soft_values.extend(state_req.soft_preferences)
    if include_exclusions:
        exclusions = collect_exclusions(snapshot, intent_result)
        soft_values.extend(f"exclude: {value}" for value in exclusions)
    hard_keys = {_norm(value) for value in hard_constraints}
    soft_preferences = tuple(
        value for value in _unique(soft_values) if _norm(value) not in hard_keys
    )

    return Requirements(
        category=category,
        hard_constraints=hard_constraints,
        soft_preferences=soft_preferences,
    )


class FusedContextualReranker:
    """Deterministic fusion reranker with an optional pointwise Qwen scorer."""

    def __init__(
        self,
        llm_ranker: Qwen3Reranker | None = None,
        *,
        top_n: int = 20,
        min_keep: int = 15,
        profile_weight: float = 0.0,
        intent_weight: float = 0.0,
        pointwise_weight: float = 0.0,
        hard_constraint_penalty: float = 0.0,
        must_not_penalty: float = 0.0,
        override_hard_penalty_multiplier: float = 2.0,
    ) -> None:
        try:
            pointwise_weight = float(os.environ["TECHJAM_POINTWISE_WEIGHT"])
        except (KeyError, TypeError, ValueError):
            pass
        try:
            intent_weight = float(os.environ["TECHJAM_INTENT_WEIGHT"])
        except (KeyError, TypeError, ValueError):
            pass
        try:
            hard_constraint_penalty = float(os.environ["TECHJAM_HARD_PENALTY"])
        except (KeyError, TypeError, ValueError):
            pass
        try:
            must_not_penalty = float(os.environ["TECHJAM_MUST_NOT_PENALTY"])
        except (KeyError, TypeError, ValueError):
            pass

        self.llm_ranker = llm_ranker
        self.top_n = top_n
        self.min_keep = min_keep
        self.profile_weight = profile_weight
        self.intent_weight = intent_weight
        self.pointwise_weight = pointwise_weight
        self.hard_constraint_penalty = hard_constraint_penalty
        self.must_not_penalty = must_not_penalty
        self.override_hard_penalty_multiplier = override_hard_penalty_multiplier
        self.fallback = LockedWeightedRrfTop10Reranker()

        self._profiles: dict[str, Mapping[str, Any]] = {}
        self._short_term: dict[str, ShortTermSummary] = {}
        self._snapshots: dict[str, Any] = {}
        self._intent_results: dict[str, Any] = {}
        self._intent_contexts: dict[str, Mapping[str, Any]] = {}
        self._override_turns: dict[str, tuple[int, ...]] = {}
        self._must_not_terms: dict[str, set[str]] = {}
        self._exclusions: dict[str, tuple[str, ...]] = {}

        self.last_fused_requirements: dict[str, Requirements] = {}
        self.last_strategy: dict[str, str] = {}
        self.last_override_active: dict[str, bool] = {}
        self.last_preselected_ids: dict[str, tuple[str, ...]] = {}
        self.last_fallback_scores: dict[str, dict[str, float]] = {}
        self.last_base_scores: dict[str, dict[str, float]] = {}
        self.last_hard_penalty_by_id: dict[str, dict[str, float]] = {}
        self.last_must_not_penalty_by_id: dict[str, dict[str, float]] = {}
        self.last_profile_boosts: dict[str, dict[str, float]] = {}
        self.last_intent_boosts: dict[str, dict[str, float]] = {}

    def set_context(
        self,
        session_id: str,
        collector: OverrideAwareRequirementsCollector,
        *,
        snapshot: Any | None = None,
        intent_result: Any | None = None,
        intent_context: Mapping[str, Any] | None = None,
    ) -> None:
        self._profiles[session_id] = merge_profile_with_snapshot(
            collector.user_profile,
            snapshot,
        )
        self._short_term[session_id] = collector.short_term_summary()
        self._snapshots[session_id] = snapshot
        self._intent_results[session_id] = intent_result
        self._intent_contexts[session_id] = dict(intent_context or {})
        self._override_turns[session_id] = tuple(getattr(collector, "override_turns", ()))
        self._exclusions[session_id] = collect_exclusions(snapshot, intent_result)
        self._must_not_terms[session_id] = {
            term
            for value in self._exclusions[session_id]
            for term in tokenize(value)
        }

    @staticmethod
    def _candidate_tokens(candidate: Candidate) -> set[str]:
        text = " ".join(
            _text(candidate.product.get(field))
            for field in VISIBLE_FIELDS
        )
        return set(tokenize(text))

    def _profile_boost(
        self,
        candidates: Mapping[str, Candidate],
        *,
        session_id: str,
    ) -> dict[str, float]:
        profile = self._profiles.get(session_id)
        features = profile_features(profile)
        if not features.preference_tags:
            return {}
        profile_terms = {
            term
            for value in features.preference_tags
            for term in tokenize(str(value))
        }
        if not profile_terms:
            return {}
        boosts: dict[str, float] = {}
        for parent_asin, candidate in candidates.items():
            overlap = len(profile_terms & self._candidate_tokens(candidate))
            if overlap:
                boosts[parent_asin] = self.profile_weight * overlap
        return boosts

    def _intent_boost(
        self,
        candidates: Mapping[str, Candidate],
        *,
        session_id: str,
    ) -> dict[str, float]:
        intent_result = self._intent_results.get(session_id)
        if intent_result is None:
            return {}
        parts = [
            str(getattr(intent_result, "semantic_query", "") or ""),
            str(getattr(intent_result, "keyword_query", "") or ""),
            str(getattr(intent_result, "normalized_query", "") or ""),
        ]
        query_terms = {term for text in parts for term in tokenize(text)}
        if not query_terms:
            return {}
        confidence = float(getattr(intent_result, "intent_confidence", 0.0) or 0.0)
        if confidence <= 0.0:
            return {}
        boosts: dict[str, float] = {}
        for parent_asin, candidate in candidates.items():
            overlap = len(query_terms & self._candidate_tokens(candidate))
            if overlap:
                boosts[parent_asin] = (
                    self.intent_weight * confidence * overlap / len(query_terms)
                )
        return boosts

    def _override_active(self, session_id: str, turn: int) -> bool:
        intent_result = self._intent_results.get(session_id)
        if bool(getattr(intent_result, "override_detected", False)):
            return True
        override_turns = self._override_turns.get(session_id, ())
        if override_turns and turn >= min(override_turns):
            return True
        debug = getattr(self._snapshots.get(session_id), "debug", None) or {}
        return bool(debug.get("category_overridden") or debug.get("intent_changed"))

    def rerank(self, candidate_set: CandidateSet, *, top_k: int) -> RerankResult:
        candidates = list(candidate_set.candidates)
        session_id = candidate_set.session_id
        if self.llm_ranker is None:
            return self.fallback.rerank(candidate_set, top_k=top_k)

        snapshot = self._snapshots.get(session_id)
        intent_result = self._intent_results.get(session_id)
        fused = fuse_requirements(
            candidate_set.requirements,
            snapshot,
            intent_result,
            override_turns=self._override_turns.get(session_id, ()),
        )
        self.last_fused_requirements[session_id] = fused
        override_active = self._override_active(session_id, candidate_set.turn)
        if override_active:
            return self.fallback.rerank(candidate_set, top_k=top_k)


        kept = tuple(candidates)
        conflicts = ()
        base_result = self.fallback.rerank(
            candidate_set,
            top_k=len(candidates),
        )
        ordered_ids = [
            ranked.parent_asin for ranked in base_result.ranked_candidates
        ]
        kept_ids = {candidate.parent_asin for candidate in kept}
        preselected_ids = [
            parent_asin for parent_asin in ordered_ids if parent_asin in kept_ids
        ]
        preselected_ids.extend(
            parent_asin for parent_asin in ordered_ids if parent_asin not in kept_ids
        )
        preselected_ids = tuple(preselected_ids[: self.top_n])

        candidate_by_id = {
            candidate.parent_asin: candidate for candidate in candidates
        }
        fallback_scores = {
            ranked.parent_asin: ranked.score
            for ranked in base_result.ranked_candidates
        }
        profile_boosts = self._profile_boost(candidate_by_id, session_id=session_id)
        intent_boosts = self._intent_boost(candidate_by_id, session_id=session_id)
        missing_by_id = {
            parent_asin: evaluate_hard_constraints(
                candidate_by_id[parent_asin],
                candidate_set.requirements,
            )
            for parent_asin in preselected_ids
        }
        plan = parse_intent(
            candidate_set.requirements.category,
            candidate_set.requirements.hard_constraints,
            candidate_set.requirements.soft_preferences,
        )
        hard_term_count = max(1, len(plan.hard_terms))
        override_multiplier = (
            self.override_hard_penalty_multiplier
            if override_active
            else 1.0
        )
        hard_penalty_by_id = {
            parent_asin: self.hard_constraint_penalty
            * override_multiplier
            * (len(missing_by_id[parent_asin]) / hard_term_count)
            for parent_asin in preselected_ids
        }
        must_not_terms = self._must_not_terms.get(session_id, set())
        must_not_penalty_by_id: dict[str, float] = {}
        if must_not_terms:
            for parent_asin in preselected_ids:
                hits = len(
                    must_not_terms & self._candidate_tokens(candidate_by_id[parent_asin])
                )
                if hits:
                    must_not_penalty_by_id[parent_asin] = self.must_not_penalty * hits

        base_scores = {
            parent_asin: (
                fallback_scores.get(parent_asin, 0.0)
                + profile_boosts.get(parent_asin, 0.0)
                + intent_boosts.get(parent_asin, 0.0)
                - hard_penalty_by_id[parent_asin]
                - must_not_penalty_by_id.get(parent_asin, 0.0)
            )
            for parent_asin in preselected_ids
        }
        preselected_sorted = tuple(
            sorted(
                preselected_ids,
                key=lambda parent_asin: (-base_scores[parent_asin], parent_asin),
            )
        )

        hard_hit_rate = (
            sum(
                1
                for parent_asin in preselected_sorted
                if not missing_by_id.get(parent_asin)
            )
            / len(preselected_sorted)
            if preselected_sorted
            else 1.0
        )
        strategy = choose_rerank_strategy(
            has_model=(self.llm_ranker is not None and self.pointwise_weight > 0.0),
            turn=candidate_set.turn,
            selected_count=len(preselected_sorted),
            top_k=top_k,
            hard_hit_rate=hard_hit_rate,
        )
        prompt_limit = min(self.top_n, max(top_k * 2, 10))
        prompt_ids = tuple(preselected_sorted[:prompt_limit])
        prompt_candidates = [
            candidate_by_id[parent_asin] for parent_asin in prompt_ids
        ]

        pointwise_scores: dict[str, float] = {}
        if self.llm_ranker is not None and self.pointwise_weight > 0.0 and strategy == "hybrid":
            try:
                pointwise_scores = self.llm_ranker.score_candidates(
                    fused,
                    prompt_candidates,
                    user_profile=self._profiles.get(session_id),
                )
            except Exception:
                pointwise_scores = {}

        if pointwise_scores:
            fallback_values = [
                fallback_scores.get(parent_asin, 0.0)
                for parent_asin in prompt_ids
            ]
            fallback_min = min(fallback_values)
            fallback_max = max(fallback_values)
            fallback_norm = {
                parent_asin: (
                    (fallback_scores.get(parent_asin, 0.0) - fallback_min)
                    / max(fallback_max - fallback_min, 1e-9)
                )
                for parent_asin in prompt_ids
            }
            fused_scores = {
                parent_asin: (
                    fallback_norm[parent_asin]
                    + self.pointwise_weight * pointwise_scores.get(parent_asin, 0.0)
                    + profile_boosts.get(parent_asin, 0.0)
                    + intent_boosts.get(parent_asin, 0.0)
                    - hard_penalty_by_id[parent_asin]
                    - must_not_penalty_by_id.get(parent_asin, 0.0)
                )
                for parent_asin in prompt_ids
            }
            ranked_ids = tuple(
                sorted(
                    prompt_ids,
                    key=lambda parent_asin: (-fused_scores[parent_asin], parent_asin),
                )[:top_k]
            )
            ranked_scores = tuple(
                pointwise_scores.get(parent_asin, fused_scores[parent_asin])
                for parent_asin in ranked_ids
            )
        else:
            ranked_ids = tuple(preselected_sorted[:top_k])
            ranked_scores = tuple(base_scores[parent_asin] for parent_asin in ranked_ids)

        self.last_strategy[session_id] = strategy
        self.last_override_active[session_id] = override_active
        self.last_preselected_ids[session_id] = preselected_sorted
        self.last_fallback_scores[session_id] = fallback_scores
        self.last_base_scores[session_id] = base_scores
        self.last_hard_penalty_by_id[session_id] = hard_penalty_by_id
        self.last_must_not_penalty_by_id[session_id] = must_not_penalty_by_id
        self.last_profile_boosts[session_id] = profile_boosts
        self.last_intent_boosts[session_id] = intent_boosts
        ranked_candidates = tuple(
            RankedCandidate(
                parent_asin=parent_asin,
                rank=rank_index,
                score=ranked_scores[rank_index - 1],
                evidence=(
                    f"base_score:{base_scores.get(parent_asin, 0.0):.6f}",
                    f"fallback_score:{fallback_scores.get(parent_asin, 0.0):.6f}",
                    f"fused_hard_missing:{','.join(missing_by_id.get(parent_asin, ())) or 'none'}",
                    f"intent_boost:{intent_boosts.get(parent_asin, 0.0):.6f}",
                    f"strategy:{strategy}",
                    f"override:{override_active}",
                ),
            )
            for rank_index, parent_asin in enumerate(ranked_ids, start=1)
        )
        return RerankResult(
            candidate_set_id=candidate_set.candidate_set_id,
            ranked_candidates=ranked_candidates,
        )


class FusedRankingAgent:
    """Opt-in agent that uses the fused reranker while keeping official retrieval."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        retrieval_mode: str = "exact",
        reranker_model: str | Path | None = None,
        use_qwen: bool = True,
        use_state_memory: bool = True,
        use_intent_router: bool = True,
        fuse_retrieval: bool = False,
    ) -> None:
        from techjam_agent.retrieval import (
            ExactDenseTop50CandidateGenerator,
            LiteTop50CandidateGenerator,
        )

        catalog_path = Path(catalog_path)
        if retrieval_mode == "exact":
            candidate_generator = ExactDenseTop50CandidateGenerator(catalog_path)
        elif retrieval_mode == "lite":
            candidate_generator = LiteTop50CandidateGenerator(catalog_path)
        else:
            raise ValueError("retrieval_mode must be 'exact' or 'lite'")

        if reranker_model is None:
            reranker_model = (
                CLEAN_ADAPTER if CLEAN_ADAPTER.is_dir() else FALLBACK_ADAPTER
            )
        llm_ranker = (
            Qwen3Reranker(str(reranker_model))
            if use_qwen and reranker_model is not None
            else None
        )
        self.candidate_generator = candidate_generator
        self.reranker = FusedContextualReranker(llm_ranker=llm_ranker)
        self.fuse_retrieval = fuse_retrieval
        state_memory_cache = Path(__file__).resolve().parent / ".cache" / "fused_state_memory_lexicon.json"
        self._sessions: dict[str, OverrideAwareRequirementsCollector] = {}
        self._state_memory = StateMemoryManager(catalog_path=catalog_path, catalog_cache_path=state_memory_cache) if use_state_memory else None
        self._intent_router = (
            IntentRouter(
                known_brands=load_catalog_brands(catalog_path),
                known_categories=load_catalog_categories(catalog_path),
            )
            if use_intent_router
            else None
        )
        self._asked_attributes: dict[str, list[str]] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = OverrideAwareRequirementsCollector(
            user_profile=user_profile
        )
        self._asked_attributes[session_id] = []
        if self._state_memory is not None:
            self._state_memory.sessions.pop(session_id, None)
        self._configure(session_id, self._sessions[session_id], None, None, None)

    def _configure(
        self,
        session_id: str,
        collector: OverrideAwareRequirementsCollector,
        snapshot: Any | None,
        intent_result: Any | None,
        intent_context: Mapping[str, Any] | None,
    ) -> None:
        self.reranker.set_context(
            session_id,
            collector,
            snapshot=snapshot,
            intent_result=intent_result,
            intent_context=intent_context,
        )

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        collector = self._sessions.get(session_id)
        if collector is None:
            raise RuntimeError("reset must be called before respond")
        collector.observe(user_message, turn)

        snapshot = None
        if self._state_memory is not None:
            snapshot = self._state_memory.update(
                session_id=session_id,
                user_id=session_id,
                utterance=user_message,
            )
        intent_result = None
        if self._intent_router is not None:
            intent_result = self._intent_router.understand(user_message)
        intent_context = intent_to_context(intent_result)
        self._configure(session_id, collector, snapshot, intent_result, intent_context)

        if turn <= 2:
            return {
                "message": "Please share any other requirements that matter.",
                "ask_attribute": "other",
                "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

        retrieval_requirements = (
            fuse_requirements(
                collector.requirements(),
                snapshot,
                intent_result,
                override_turns=collector.override_turns,
            )
            if self.fuse_retrieval
            else collector.requirements()
        )
        candidate_set = self.candidate_generator.generate(
            retrieval_requirements,
            session_id=session_id,
            turn=turn,
        )

        if self._state_memory is not None:
            try:
                snapshot = self._state_memory.apply_retrieval_feedback(
                    session_id=session_id,
                    user_id=session_id,
                    query=user_message,
                    candidate_count=len(candidate_set.candidates),
                )
                self._configure(
                    session_id,
                    collector,
                    snapshot,
                    intent_result,
                    intent_context,
                )
            except Exception:
                pass

        result = self.reranker.rerank(candidate_set, top_k=top_k)
        result.validate_against(candidate_set, top_k=top_k)
        return {
            "message": "Here are the best matches for all requirements you shared.",
            "ask_attribute": None,
            "recommendations": [
                {"parent_asin": candidate.parent_asin}
                for candidate in result.ranked_candidates
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


__all__ = [
    "FusedContextualReranker",
    "FusedRankingAgent",
    "FusedRequirements",
    "collect_exclusions",
    "fuse_requirements",
]

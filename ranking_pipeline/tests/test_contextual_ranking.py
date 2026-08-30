from __future__ import annotations

import unittest

from techjam_agent.contracts import Candidate, CandidateSet, RankedCandidate, Requirements
from ranking_pipeline.context import parse_override_message
from ranking_pipeline.contextual_ranking import (
    CandidatePoolMetrics,
    ConstraintConflict,
    HybridContextualReranker,
    RecommendationClarificationPolicy,
    assess_candidate_pool,
    choose_clarification_attribute,
    choose_rerank_strategy,
    filter_hard_conflicts,
)
from ranking_pipeline.prompt import LLMRankResult, build_rerank_prompt, parse_rerank_output


def make_candidate(index: int, *, title: str = "", features: list[str] | None = None) -> Candidate:
    return Candidate(
        parent_asin=f"P{index:02d}",
        candidate_rank=index + 1,
        source_ranks={"route": index + 1},
        product={
            "title": title or f"Cotton blue shirt {index}",
            "categories": ["Clothing", "Women's Shirts"],
            "features": features or ["cotton", "blue", "machine washable"],
            "details": {"Color": "Blue"},
            "description": [],
            "store": "Example",
        },
    )


def make_candidate_set(requirements: Requirements, count: int = 50) -> CandidateSet:
    return CandidateSet(
        candidate_set_id="session:3",
        session_id="session",
        turn=3,
        requirements=requirements,
        candidates=tuple(make_candidate(index) for index in range(count)),
    )


class FakeLLMRanker:
    def __init__(self, result: LLMRankResult | None = None, error: bool = False) -> None:
        self.result = result
        self.error = error

    def rank(self, prompt: str, *, allowed_ids, top_k: int) -> LLMRankResult:
        del prompt, allowed_ids, top_k
        if self.error:
            raise RuntimeError("model unavailable")
        assert self.result is not None
        return self.result


class ContextualRankingTests(unittest.TestCase):
    def test_hard_filter_removes_only_a_fully_conflicted_candidate(self) -> None:
        requirements = Requirements("Women's Shirts", ("cotton", "blue"), ())
        candidates = [make_candidate(index) for index in range(49)]
        candidates.append(
            Candidate(
                parent_asin="P49",
                candidate_rank=50,
                source_ranks={"route": 50},
                product={
                    "title": "Polyester red dress",
                    "categories": ["Clothing", "Dresses"],
                    "features": ["polyester", "red"],
                    "details": {},
                    "description": [],
                    "store": "Example",
                },
            )
        )

        kept, conflicts = filter_hard_conflicts(candidates, requirements, min_keep=15)

        self.assertEqual(len(kept), 49)
        self.assertEqual([conflict.parent_asin for conflict in conflicts], ["P49"])
        self.assertEqual(conflicts[0].missing_terms, ("cotton", "blue"))

    def test_prompt_is_compact_and_contains_constraints(self) -> None:
        requirements = Requirements(
            "Women's Shirts",
            ("cotton", "color: blue"),
            ("lightweight",),
        )
        candidates = [make_candidate(index) for index in range(5)]

        prompt = build_rerank_prompt(
            requirements,
            candidates,
            top_k=3,
            user_profile={"preference_tags": ["fit", "comfort"]},
        )

        self.assertIn("cotton", prompt)
        self.assertIn("color: blue", prompt)
        self.assertIn("lightweight", prompt)
        self.assertIn("fit", prompt)
        self.assertIn('"ranked_ids"', prompt)
        self.assertIn("P00", prompt)

    def test_parse_rerank_output_accepts_code_fence_and_filters_ids(self) -> None:
        text = (
            "```json\n"
            '{"ranked_ids":["P03","P01","OUTSIDE"],"constraint_conflicts":["no blue"],'
            '"confidence":0.86,"need_clarification":false}\n'
            "```"
        )

        result = parse_rerank_output(
            text,
            allowed_ids=["P01", "P02", "P03"],
            top_k=2,
        )

        self.assertEqual(result.ranked_ids, ("P03", "P01"))
        self.assertEqual(result.constraint_conflicts, ("no blue",))
        self.assertAlmostEqual(result.confidence, 0.86)
        self.assertFalse(result.need_clarification)

    def test_hybrid_reranker_falls_back_when_local_model_fails(self) -> None:
        candidate_set = make_candidate_set(
            Requirements("Women's Shirts", ("cotton", "blue"), ())
        )
        reranker = HybridContextualReranker(
            llm_ranker=FakeLLMRanker(error=True),
            top_n=20,
        )

        result = reranker.rerank(candidate_set, top_k=10)

        result.validate_against(candidate_set, top_k=10)
        self.assertIsNone(reranker.last_model_result)
        self.assertEqual(len(result.ranked_candidates), 10)
        self.assertEqual(result.ranked_candidates[0].parent_asin, "P00")

    def test_hybrid_reranker_uses_local_llm_ranking_when_available(self) -> None:
        candidate_set = make_candidate_set(
            Requirements("Women's Shirts", ("cotton", "blue"), ())
        )
        expected = LLMRankResult(("P02", "P01"), (), 0.91, False)
        reranker = HybridContextualReranker(
            llm_ranker=FakeLLMRanker(expected),
            top_n=20,
        )

        result = reranker.rerank(candidate_set, top_k=2)

        self.assertEqual(
            [candidate.parent_asin for candidate in result.ranked_candidates],
            ["P02", "P01"],
        )
        self.assertIs(reranker.last_model_result, expected)

    def test_policy_asks_when_confidence_or_conflicts_are_low(self) -> None:
        policy = RecommendationClarificationPolicy()
        ranked = [
            RankedCandidate("P00", 1, 1.0, ("model:test",)),
            RankedCandidate("P01", 2, 0.8, ("model:test",)),
        ]

        low_confidence = LLMRankResult(("P00", "P01"), (), 0.40, False)
        decision = policy.decide(
            ranked,
            model_result=low_confidence,
            conflicts=(),
        )
        self.assertEqual(decision.action, "clarify")

        conflict_decision = policy.decide(
            ranked,
            model_result=LLMRankResult(("P00", "P01"), (), 0.90, False),
            conflicts=[ConstraintConflict("P01", ("cotton",))],
        )
        self.assertEqual(conflict_decision.action, "clarify")

    def test_override_message_is_parsed(self) -> None:
        values = parse_override_message(
            "Actually, ignore my earlier preference. What I need is: cotton; blue."
        )
        self.assertEqual(values, ("cotton", "blue"))

    def test_parse_output_extracts_scores_and_reasons(self) -> None:
        text = (
            '{"ranked_ids":["P02","P01"],"scores":[0.9,0.7],'
            '"reasons":["exact cotton","blue match"],'
            '"constraint_conflicts":[],"confidence":0.9,'
            '"need_clarification":false}'
        )
        result = parse_rerank_output(
            text,
            allowed_ids=["P01", "P02"],
            top_k=2,
        )
        self.assertEqual(result.scores, (0.9, 0.7))
        self.assertEqual(result.reasons, ("exact cotton", "blue match"))

    def test_dynamic_strategy_and_clarification_selection(self) -> None:
        self.assertEqual(choose_rerank_strategy(has_model=False, turn=3, selected_count=20, top_k=10, hard_hit_rate=0.5), "hybrid")
        self.assertEqual(choose_rerank_strategy(has_model=True, turn=10, selected_count=20, top_k=10, hard_hit_rate=0.5), "locked")
        self.assertEqual(choose_rerank_strategy(has_model=True, turn=3, selected_count=10, top_k=10, hard_hit_rate=1.0), "hybrid")
        self.assertEqual(choose_rerank_strategy(has_model=True, turn=3, selected_count=20, top_k=10, hard_hit_rate=0.5), "hybrid")

        self.assertEqual(
            choose_clarification_attribute(
                Requirements("Women's Shirts", ("cotton",), ()),
                asked_attributes=("color",),
                over_general=True,
            ),
            "material",
        )

    def test_policy_uses_over_general_and_asked_attributes(self) -> None:
        policy = RecommendationClarificationPolicy()
        ranked = [
            RankedCandidate("P00", 1, 0.8, ("model:test",)),
            RankedCandidate("P01", 2, 0.79, ("model:test",)),
        ]
        metrics = CandidatePoolMetrics(True, 0.8, 0.79, 0.01, 0.5, 20, "local")
        decision = policy.decide(
            ranked,
            model_result=LLMRankResult(("P00", "P01"), (), 0.8, False, (0.8, 0.79), ("a", "b")),
            conflicts=(),
            requirements=Requirements("Women's Shirts", ("cotton",), ()),
            asked_attributes=("material",),
            pool_metrics=metrics,
        )
        self.assertEqual(decision.action, "clarify")
        self.assertEqual(decision.ask_attribute, "color")


if __name__ == "__main__":
    unittest.main()

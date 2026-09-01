"""Behavioral integration acceptance using real lexical indexes on a tiny catalog."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shopping_agent import FinalAgent
from shopping_agent.retrieval import RetrievalRequest
from shopping_agent.policy import PreRetrievalPolicy
from intent_router.turn_router import TurnIntentRouter


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.catalog = Path(cls.tmp.name) / "catalog.jsonl"
        rows = [
            ("BLACK", "black cotton dress", ["Clothing", "Dresses"], 30),
            ("BLUE", "blue cotton dress", ["Clothing", "Dresses"], 40),
            ("EXPENSIVE", "blue cotton dress", ["Clothing", "Dresses"], 100),
            ("UNKNOWN", "blue cotton dress", ["Clothing", "Dresses"], None),
            ("SHOE", "canvas shoes comfortable", ["Shoes"], 25),
            ("LEATHER", "leather shoes", ["Shoes"], 20),
        ]
        cls.catalog.write_text("\n".join(json.dumps({"parent_asin": asin, "title": title, "categories": cats,
            "price": price, "store": "Acme", "features": [], "details": {}, "description": [], "rating_number": 10}) for asin, title, cats, price in rows))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.agent = FinalAgent(self.catalog, trace_enabled=True)
        self.agent.reset("s", {"preference_tags": ["comfort"]})

    def respond(self, text, turn=1):
        result = self.agent.respond("s", text, turn, 10)
        self.assertEqual(self.agent.errors, [])
        return result

    def event(self, stage):
        return next(e["output"] for e in self.agent.trace[-1]["events"] if e["stage"] == stage)

    def test_real_pipeline_variable_candidates_and_budget(self):
        result = self.respond("I need a blue dress under $50.")
        self.assertEqual([r["parent_asin"] for r in result["recommendations"]], ["BLUE"])
        self.assertEqual(self.event("2_retrieval")["returned_count"], 1)
        self.assertEqual(self.event("3_feedback")["shown_asins"], ("BLUE",))

    def test_multiturn_override_clear_and_category_switch(self):
        self.respond("I need a black dress under $50.")
        self.respond("Blue instead.", 2)
        hard = self.event("3_state")["hard_constraints"]
        self.assertEqual(hard, {"category": "dress", "color": "blue", "price_max": 50.0})
        self.respond("No budget limit.", 3)
        self.assertNotIn("price_max", self.event("3_state")["hard_constraints"])
        result = self.respond("Switch to shoes, not leather.", 4)
        self.assertEqual(self.event("3_state")["hard_constraints"], {"category": "shoes"})
        self.assertEqual(self.event("3_state")["exclusions"], {"material": ["leather"]})
        self.assertEqual([r["parent_asin"] for r in result["recommendations"]], ["SHOE"])
        self.assertTrue(self.event("3_state")["suggestions"]["category_changed"])

    def test_pre_policy_skips_retrieval_and_records_one_actual_question(self):
        result = self.respond("Help me find something.")
        self.assertEqual(result["ask_attribute"], "category")
        self.assertNotIn("2_retrieval", [e["stage"] for e in self.agent.trace[-1]["events"]])
        feedback = self.event("3_feedback")
        self.assertEqual(feedback["suggestions"]["clarification_count"], 1)
        self.assertEqual(feedback["pending_question"]["target_slot"], "category")
        self.respond("A black dress under $50.", 2)
        self.assertEqual(self.event("3_state")["state_version"], 3)
        self.assertEqual(self.event("3_feedback")["suggestions"]["clarification_count"], 1)

    def test_empty_pool_never_padded_and_question_limit(self):
        for turn in range(1, 11):
            response = self.respond("I need a purple dress under $1.", turn)
            self.assertEqual(response["recommendations"], [])
            if turn >= 3:
                self.assertIsNone(response["ask_attribute"])
        self.assertEqual(self.event("3_feedback")["suggestions"]["clarification_count"], 2)
        with self.assertRaises(ValueError):
            self.agent.respond("s", "again", 11, 10)

    def test_idempotency_and_invalid_order(self):
        first = self.respond("I need a blue dress under $50.")
        self.assertEqual(self.agent.respond("s", "I need a blue dress under $50.", 1, 10), first)
        self.assertEqual(self.agent.memory.versions["s"], 2)
        with self.assertRaises(ValueError):
            self.agent.respond("s", "other input", 1, 10)
        with self.assertRaises(ValueError):
            self.agent.respond("s", "other input", 3, 10)

    def test_session_isolation(self):
        self.respond("I need a blue dress under $50.")
        self.agent.reset("other", {})
        out = self.agent.respond("other", "Hello", 1)
        self.assertEqual(out["ask_attribute"], "category")
        self.assertEqual(self.agent.memory.sessions["s"].hard_slots["color"].value, "blue")

    def test_exclusion_can_be_removed_without_setting_positive_slot(self):
        self.respond("I need a dress, not black.")
        self.respond("Black is also fine.", 2)
        state = self.event("3_state")
        self.assertEqual(state["exclusions"], {})
        self.assertNotIn("color", state["hard_constraints"])

    def test_disclosed_feature_does_not_switch_category(self):
        self.respond("I need a dress.")
        self.respond("For that, what matters is: a gift for kids inspired by Goddess.", 2)
        state = self.event("3_state")
        self.assertEqual(state["hard_constraints"], {"category": "dress"})
        self.assertTrue(any(k.startswith("feature_") for k in state["soft_preferences"]))
        self.assertNotIn("brand", state["soft_preferences"])

    def test_failure_does_not_recommend_stale_candidates(self):
        self.respond("I need a black dress.")
        with patch.object(self.agent.reranker, "rerank", side_effect=RuntimeError("test failure")):
            result = self.agent.respond("s", "Blue instead.", 2, 10)
        self.assertEqual(result["recommendations"], [])
        self.assertEqual(self.agent.errors[-1]["stage"], "4B_ranking")
        self.assertEqual(self.event("3_feedback")["hard_constraints"]["color"], "blue")

    def test_cross_session_result_rejected(self):
        self.respond("I need a dress.")
        state = self.agent.memory.snapshot("s")
        result = self.agent.retriever.generate(RetrievalRequest.from_state(state))
        with self.assertRaises(ValueError):
            result.validate_context(session_id="wrong", turn=state.turn, state_version=state.state_version)

    def test_soft_retry_is_bounded_and_does_not_drop_hard_constraints(self):
        self.respond("I need a purple dress, preferably cotton.")
        events = self.agent.trace[-1]["events"]
        self.assertEqual(sum(e["stage"] == "2_retrieval_retry" for e in events), 1)
        retry = self.event("5_retry_plan")
        self.assertTrue(retry["relax_soft"])
        self.assertEqual(retry["state"]["hard_constraints"]["color"], "purple")

    def test_router_initial_natural_phrase_not_whole_category(self):
        parsed = TurnIntentRouter().understand_turn("I'm looking for a black dress under $50.")
        self.assertEqual(next(op.values[0] for op in parsed.slot_updates if op.slot == "category"), "dress")

    def test_short_answers_and_structured_override_reach_state(self):
        router = TurnIntentRouter()
        feature = router.understand_turn("Drawstring closure.", pending_question={"target_slot": "feature", "constraint_type": "soft"})
        self.assertTrue(any(op.slot.startswith("feature_") and op.operation == "set" for op in feature.slot_updates))
        budget = router.understand_turn("$50", pending_question={"target_slot": "budget", "constraint_type": "hard"})
        self.assertTrue(any(op.slot == "price_max" and op.values == (50.0,) for op in budget.slot_updates))
        self.respond("I'm looking for dresses. Cotton.")
        self.respond("Actually, what I need is: linen.", 2)
        state = self.event("3_state")
        self.assertEqual(state["hard_constraints"]["material"], "linen")
        self.assertNotIn("cotton", json.dumps(state["hard_constraints"]))

    def test_catalog_store_words_are_not_accidental_brands(self):
        self.agent.router.known_brands.update({"switch", "not", "need"})
        self.respond("Switch to shoes, not leather.")
        self.assertEqual(self.event("3_state")["hard_constraints"], {"category": "shoes"})

    def test_explicit_brand_is_passed_to_filter(self):
        self.respond("I need a dress from Acme.")
        self.assertEqual(self.event("3_state")["hard_constraints"]["brand"], "acme")

    def test_official_preference_override_does_not_leave_stale_hard_material(self):
        self.respond("I'm looking for Shoes. Material: leather.")
        self.respond("Actually, ignore my earlier preference. What I need is: canvas.", 2)
        state = self.event("3_state")
        self.assertNotIn("leather", json.dumps(state["hard_constraints"]))
        self.assertEqual(state["soft_preferences"], {})
        self.assertIn("canvas", json.dumps(state["hard_constraints"]))

    def test_post_policy_can_clarify_broad_pool(self):
        self.respond("I need a dress.")
        state = self.agent.memory.snapshot("s")
        result = self.agent.retriever.generate(RetrievalRequest.from_state(state))
        result = replace(result, stats=replace(result.stats, matched_count=150, filtered_count=150))
        ranking = self.agent.reranker.rerank(result, top_k=10)
        decision = self.agent.post_policy.decide(state, result, ranking)
        self.assertEqual(decision.action, "clarify")
        self.assertEqual(decision.question["target_slot"], "feature")

    def test_score_compat_policy_is_state_based_and_configuration_is_explicit(self):
        self.respond("I need a blue dress under $50.")
        state = self.agent.memory.snapshot("s")
        decision = PreRetrievalPolicy(minimum_evidence=4).decide(state)
        self.assertEqual(decision.action, "clarify")
        self.assertEqual(decision.question["target_slot"], "other")
        score_agent = FinalAgent(self.catalog, orchestration_mode="score_compat")
        self.assertEqual(score_agent.retriever.mode, "recall_compat")
        self.assertEqual(score_agent.reranker.mode, "locked")

    def test_clarification_ablation_profiles_expose_distinct_question_budgets(self):
        state_based = FinalAgent(self.catalog, orchestration_mode="score_compat")
        fixed = FinalAgent(self.catalog, orchestration_mode="score_compat", clarification_mode="fixed_two_dynamic")
        value_based = FinalAgent(self.catalog, orchestration_mode="score_compat", clarification_mode="one_then_value")
        self.assertEqual((state_based.pre_policy.minimum_questions, state_based.pre_policy.minimum_evidence,
                          state_based.post_policy.max_questions), (0, 4, 2))
        self.assertEqual((fixed.pre_policy.minimum_questions, fixed.pre_policy.minimum_evidence,
                          fixed.post_policy.max_questions), (2, 0, 3))
        self.assertEqual((value_based.pre_policy.minimum_questions, value_based.pre_policy.minimum_evidence,
                          value_based.post_policy.max_questions), (1, 4, 3))

    def test_profile_is_soft_and_decay_does_not_touch_hard(self):
        self.respond("I need a blue dress, ideally cotton.")
        before = self.event("3_state")
        self.assertEqual(before["hard_constraints"]["color"], "blue")
        self.respond("Thanks.", 2)
        after = self.event("3_state")
        self.assertEqual(after["hard_constraints"], before["hard_constraints"])
        self.assertLess(after["soft_preferences"]["material"][0]["weight"], before["soft_preferences"]["material"][0]["weight"])
        self.assertEqual(after["profile_hints"]["preference_tags"], ["comfort"])

    def test_negative_feedback_gets_a_new_question_without_repeating_feature(self):
        self.respond("I need a dress.")
        first = self.respond("Those options are not quite right yet.", 2)
        self.assertEqual(first["ask_attribute"], "feature")
        self.respond("For that, what matters is: cotton.", 3)
        second = self.respond("Those options are not quite right yet.", 4)
        self.assertEqual(second["ask_attribute"], "other")
        self.assertEqual(self.event("3_feedback")["suggestions"]["clarification_count"], 2)

    def test_dense_runtime_failure_keeps_lexical_candidates_and_warning(self):
        backend = self.agent.retriever.backend
        with patch.object(backend, "_dense_ranking", side_effect=RuntimeError("model offline"), create=True):
            result = self.respond("I need a black dress under $50.")
        self.assertEqual(result["recommendations"][0]["parent_asin"], "BLACK")
        self.assertTrue(any("Dense failed" in warning for warning in self.event("2_retrieval")["warnings"]))


if __name__ == "__main__":
    unittest.main()

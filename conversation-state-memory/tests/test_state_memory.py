import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from state_memory import NextAction, Route, StateMemoryManager


class StateMemoryManagerTests(unittest.TestCase):
    def setUp(self):
        self.manager = StateMemoryManager()

    def update(self, text, feedback=None):
        return self.manager.update("s1", "u1", text, feedback)

    def test_accumulates_hard_and_soft_constraints(self):
        self.update("I need a dress for work")
        self.update("black and under $50")
        snapshot = self.update("something slimming")
        self.assertEqual(snapshot.must_match["category"], "dress")
        self.assertEqual(snapshot.must_match["occasion"], "work")
        self.assertEqual(snapshot.must_match["color"], "black")
        self.assertEqual(snapshot.must_match["price_max"], 50.0)
        self.assertIn("slimming", snapshot.should_match["style"])

    def test_replaces_color_and_keeps_explicit_rejection(self):
        self.update("show me a black dress")
        snapshot = self.update("not black, blue instead")
        self.assertEqual(snapshot.must_match["color"], "blue")
        self.assertIn("black", snapshot.must_not_match["color"])

    def test_category_override_reroutes(self):
        self.update("I need a work dress under $50")
        snapshot = self.update("actually show me running shoes")
        self.assertEqual(snapshot.must_match["category"], "shoes")
        self.assertEqual(snapshot.action, NextAction.REROUTE)

    def test_budget_is_rewritten(self):
        self.update("a dress under $50")
        snapshot = self.update("I can go up to $80")
        self.assertEqual(snapshot.must_match["price_max"], 80.0)

    def test_profile_does_not_override_current_hard_constraint(self):
        self.update("I need a black dress")
        self.update("another black work dress")
        snapshot = self.manager.update("s2", "u1", "show me a blue dress")
        self.assertEqual(snapshot.must_match["color"], "blue")
        self.assertNotIn("color", snapshot.profile_hints)

    def test_overload_asks_high_information_question(self):
        snapshot = self.update("show me something nice", {"candidate_count": 1000})
        self.assertEqual(snapshot.action, NextAction.ASK_CLARIFICATION)
        self.assertIn("type of item", snapshot.clarification_question)

    def test_browsing_and_buying_routes(self):
        snapshot = self.update("what bags are trending this season")
        self.assertEqual(snapshot.route, Route.BROWSING_DENSE)
        snapshot = self.update("show me a black tote under $80")
        self.assertEqual(snapshot.route, Route.BUYING_FILTER)

    def test_zero_results_preserves_hard_constraints_and_relaxes_soft_ones(self):
        self.update("I need a slimming blue dress under $80")
        snapshot = self.update("please search again", {"candidate_count": 0})
        self.assertEqual(snapshot.must_match["category"], "dress")
        self.assertEqual(snapshot.must_match["color"], "blue")
        self.assertEqual(snapshot.debug["relax_soft_preferences"], ["style"])

    def test_soft_preference_weight_decays_over_turns(self):
        first = self.update("I need a minimal dress")
        first_weight = first.should_match["style"]["minimal"]
        self.update("for work")
        later = self.update("under $80")
        self.assertLess(later.should_match["style"]["minimal"], first_weight)

    def test_conversion_signal(self):
        self.update("show me a dress")
        snapshot = self.update("I'll take this")
        self.assertEqual(snapshot.action, NextAction.CONVERT)

    def test_comparison_action_is_not_sticky_on_follow_up(self):
        self.update("compare these dresses")
        snapshot = self.update("blue please")
        self.assertEqual(snapshot.action, NextAction.RETRIEVE_BUYING)

    def test_overload_stops_prompting_when_all_priority_slots_are_present(self):
        self.update("show me a blue work dress under $80", {"candidate_count": 1000})
        snapshot = self.update("women size M", {"candidate_count": 1000})
        self.assertNotEqual(snapshot.action, NextAction.ASK_CLARIFICATION)


if __name__ == "__main__":
    unittest.main()

"""Ablation variant: rank starting at turn 2 instead of turn 3.

The original ``RankingAgent`` forces two clarification turns (turn 1 and turn 2)
before it ever runs retrieval/ranking.  This subclass keeps the rest of the
pipeline identical and only changes the mandatory clarification boundary to
``turn <= 1``, so the agent asks once and starts ranking on the second turn.
"""

from __future__ import annotations

from ranking_pipeline.agent import RankingAgent


class RankAfterFirstAgent(RankingAgent):
    """Same ranking pipeline, but only turn 1 is a mandatory clarification."""

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
        snapshot = self._update_external_context(session_id, user_message)
        self._configure_reranker(session_id, collector)
        pre_retrieval = self._pre_retrieval_decision(session_id, snapshot)
        if pre_retrieval is not None:
            return pre_retrieval
        if turn <= 1:
            self._record_asked(session_id, "other")
            return {
                "message": "Please share any other requirements that matter.",
                "ask_attribute": "other",
                "recommendations": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
        candidate_set = self.candidate_generator.generate(
            collector.requirements(), session_id=session_id, turn=turn
        )
        result = self.reranker.rerank(candidate_set, top_k=top_k)
        result.validate_against(candidate_set, top_k=top_k)
        if self._policy_enabled and turn < 10:
            decision = self._policy_decision(result, session_id)
            if decision is not None and decision.action == "clarify":
                self._record_asked(session_id, decision.ask_attribute)
                return {
                    "message": decision.message,
                    "ask_attribute": decision.ask_attribute,
                    "recommendations": [],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                }
        return {
            "message": "Here are the best matches for all requirements you shared.",
            "ask_attribute": None,
            "recommendations": [
                {"parent_asin": candidate.parent_asin}
                for candidate in result.ranked_candidates
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


__all__ = ["RankAfterFirstAgent"]

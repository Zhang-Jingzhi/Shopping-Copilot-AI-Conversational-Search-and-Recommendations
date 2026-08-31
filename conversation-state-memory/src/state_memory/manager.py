from __future__ import annotations

from .context_program import ContextProgrammer
from .extractor import RuleBasedExtractor
from .models import ContextSnapshot, SessionState, UserProfile
from .profile import ProfileDistiller
from .state_machine import DynamicStateMachine


class StateMemoryManager:
    """In-memory state manager; persist sessions/profiles outside this class if needed."""

    def __init__(self) -> None:
        self.sessions: dict[str, SessionState] = {}
        self.profiles: dict[str, UserProfile] = {}
        self.extractor = RuleBasedExtractor()
        self.state_machine = DynamicStateMachine()
        self.profile_distiller = ProfileDistiller()
        self.context_programmer = ContextProgrammer()

    def update(
        self,
        session_id: str,
        user_id: str,
        utterance: str,
        retrieval_feedback: dict | None = None,
    ) -> ContextSnapshot:
        state = self.sessions.setdefault(session_id, SessionState(session_id=session_id))
        profile = self.profiles.setdefault(user_id, UserProfile(user_id=user_id))
        if retrieval_feedback:
            state.candidate_count = retrieval_feedback.get("candidate_count", state.candidate_count)
            state.shown_asins.extend(retrieval_feedback.get("shown_asins", []))
        extraction = self.extractor.extract(utterance)
        delta = self.state_machine.apply(state, extraction, utterance)
        self.profile_distiller.update(profile, extraction)
        debug = {
            "added_slots": delta.added_slots,
            "updated_slots": delta.updated_slots,
            "erased_slots": delta.erased_slots,
            "rejected_values": delta.rejected_values,
            "intent_changed": delta.intent_changed,
            "category_overridden": delta.category_overridden,
            "candidate_count": state.candidate_count,
            "relax_soft_preferences": (
                list(state.soft_slots) if state.candidate_count == 0 else []
            ),
        }
        return self.context_programmer.build(utterance, state, profile, debug)

    def apply_retrieval_feedback(
        self,
        session_id: str,
        user_id: str,
        query: str,
        candidate_count: int,
        shown_asins: list[str] | None = None,
    ) -> ContextSnapshot:
        """Reprogram the current turn after retrieval, without extracting it twice."""
        state = self.sessions[session_id]
        profile = self.profiles[user_id]
        state.candidate_count = candidate_count
        if shown_asins:
            state.shown_asins.extend(asin for asin in shown_asins if asin not in state.shown_asins)
        debug = {
            "added_slots": [],
            "updated_slots": [],
            "erased_slots": [],
            "rejected_values": [],
            "intent_changed": False,
            "category_overridden": False,
            "candidate_count": candidate_count,
            "relax_soft_preferences": list(state.soft_slots) if candidate_count == 0 else [],
        }
        return self.context_programmer.build(query, state, profile, debug)

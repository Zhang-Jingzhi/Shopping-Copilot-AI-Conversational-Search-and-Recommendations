"""Official reset/respond API and observable, versioned orchestration."""
from copy import deepcopy
from time import perf_counter

from intent_router.turn_router import TurnIntentRouter
from intent_router.catalog_lexicon import NON_BRAND_STORE_WORDS
from state_memory.structured_manager import StructuredStateMemoryManager
from .retrieval import RetrievalRequest, StateAwareRetriever
from .ranking import StateAwareReranker
from .policy import PreRetrievalPolicy, PostRetrievalPolicy


class FinalAgent:
    def __init__(self, catalog_path, *, retrieval_backend=None, ranking_mode=None, orchestration_mode="adaptive", clarification_mode=None, trace_enabled=False):
        if orchestration_mode not in {"adaptive", "score_compat"}:
            raise ValueError("orchestration_mode must be adaptive or score_compat")
        if ranking_mode is None:
            ranking_mode = "locked" if orchestration_mode == "score_compat" else "hybrid"
        if clarification_mode is None:
            clarification_mode = "state_evidence" if orchestration_mode == "score_compat" else "strict_dynamic"
        profiles = {
            "strict_dynamic": (0, 0, 2),
            "state_evidence": (4, 0, 2),
            "fixed_two_dynamic": (0, 2, 3),
            "one_then_value": (4, 1, 3),
        }
        if clarification_mode not in profiles:
            raise ValueError("unsupported clarification_mode")
        self.orchestration_mode = orchestration_mode
        self.clarification_mode = clarification_mode
        self.retriever = StateAwareRetriever(catalog_path, backend=retrieval_backend,
                                             mode="recall_compat" if orchestration_mode == "score_compat" else "strict")
        brands = {str(p.get("store") or "").strip().lower() for p in self.retriever.products.values()}
        self.router = TurnIntentRouter(known_brands={b for b in brands if len(b) >= 3 and b not in NON_BRAND_STORE_WORDS})
        self.memory = StructuredStateMemoryManager()
        minimum_evidence, minimum_questions, max_questions = profiles[clarification_mode]
        self.pre_policy = PreRetrievalPolicy(minimum_evidence, minimum_questions, max_questions)
        self.reranker = StateAwareReranker(ranking_mode)
        self.post_policy = PostRetrievalPolicy(max_questions)
        self.calls = {}
        self.trace_enabled = trace_enabled
        self.trace = []
        self.errors = []

    def reset(self, session_id, user_profile):
        self.memory.reset(session_id, user_profile)
        self.calls[session_id] = {}

    def respond(self, session_id, user_message, turn, top_k=10):
        if session_id not in self.calls:
            raise ValueError("reset(session_id, user_profile) is required")
        if type(turn) is not int or not 1 <= turn <= 10:
            raise ValueError("turn must be between 1 and 10")
        if type(top_k) is not int or not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        calls = self.calls[session_id]
        if turn in calls:
            previous = calls[turn]
            if (user_message, top_k) != previous[:2]:
                raise ValueError("conflicting replay of the same turn")
            return deepcopy(previous[2])
        if turn != len(calls) + 1:
            raise ValueError("turns must be sequential")
        started = perf_counter()
        events = []

        def observe(stage, payload):
            events.append({"stage": stage, "elapsed_ms": round((perf_counter() - started) * 1000, 3), "output": payload})

        state = None
        retrieval = None
        decision = None
        stage = "1_intent"
        try:
            intent_result = self.router.understand_turn(user_message, pending_question=self.memory.pending[session_id])
            handoff = intent_result.to_state_handoff(session_id=session_id, turn=turn)
            observe("1_intent", handoff)  # BP1: intent_result, handoff
            stage = "3_state"
            state = self.memory.update_from_intent(handoff)
            observe("3_state", state.to_dict())  # BP3: accumulated state
            stage = "4A_pre_policy"
            pre_decision = self.pre_policy.decide(state)
            observe("4A_pre_policy", pre_decision.to_dict())  # BP4A
            if pre_decision.action == "clarify":
                decision = pre_decision
                ranking = None
            else:
                stage = "2_retrieval"
                request = RetrievalRequest.from_state(state)
                observe("5_plan", request.to_dict())  # BP5: adaptive plan
                retrieval = self.retriever.generate(request)
                retrieval.validate_context(session_id=session_id, turn=turn, state_version=state.state_version)
                observe("2_retrieval", retrieval.to_dict())  # BP2: candidates + filter counts
                if not retrieval.candidates and state.soft_preferences:
                    request = request.retry()  # Bounded re-orchestration; never relax hard constraints.
                    observe("5_retry_plan", request.to_dict())
                    retrieval = self.retriever.generate(request)
                    retrieval.validate_context(session_id=session_id, turn=turn, state_version=state.state_version)
                    observe("2_retrieval_retry", retrieval.to_dict())
                stage = "4B_ranking"
                ranking = self.reranker.rerank(retrieval, top_k=top_k)
                ranking.validate_against(retrieval, top_k=top_k)
                observe("4B_ranking", ranking.to_dict())  # BP4B-rank
                stage = "4B_post_policy"
                decision = self.post_policy.decide(state, retrieval, ranking)
                observe("4B_post_policy", decision.to_dict())  # BP4B-policy
            recommendations = [] if decision.action == "clarify" or ranking is None else [
                {"parent_asin": row.parent_asin, "score": row.score} for row in ranking.ranked_candidates]
            response = {
                "message": decision.question["message"] if decision.question else ("Here are ranked catalog candidates. Please check their product details." if recommendations else "No candidates passed the current catalog checks."),
                "ask_attribute": decision.question["ask_attribute"] if decision.question else None,
                "recommendations": recommendations,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
        except Exception as exc:
            error = {"session_id": session_id, "turn": turn, "stage": stage, "type": type(exc).__name__, "message": str(exc)}
            self.errors.append(error)
            observe("error", error)
            # Fail closed: do not reuse stale candidates or drop hard constraints.
            decision = None
            response = {"message": "I could not complete this search. Please restate your requirements.", "ask_attribute": None, "recommendations": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
        observe("response", response)  # BP-response
        if state is not None:
            feedback = self.memory.record_execution(session_id, turn=turn,
                question=decision.question if decision else None,
                shown_asins=[row["parent_asin"] for row in response["recommendations"]],
                candidate_count=retrieval.returned_count if retrieval is not None else None)
            observe("3_feedback", feedback.to_dict())  # BP-feedback: count actual questions/shown ASINs
        calls[turn] = (user_message, top_k, deepcopy(response))
        if self.trace_enabled:
            self.trace.append({"session_id": session_id, "turn": turn, "user_message": user_message, "events": events})
        return response

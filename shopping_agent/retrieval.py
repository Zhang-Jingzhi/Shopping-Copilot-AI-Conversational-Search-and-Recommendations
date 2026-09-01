"""State-aware adapter over module 2's real indexes.

Strict mode filters and truncates a variable pool. The submitted recall-compatible
mode preserves module 2's deterministic Top-50 candidate contract.
"""
from dataclasses import asdict, dataclass, replace
import math

from techjam_agent.contracts import Candidate, PRODUCT_FIELDS, Requirements
from techjam_agent.contracts_v2 import RetrievalResultV2, RetrievalStats
from techjam_agent.query import tokenize, parse_text
from techjam_agent.retrieval import LiteTop50CandidateGenerator, build_retrieval_plan, _text
from state_memory.contracts import StateSnapshotV2


def sequence(value):
    return value if isinstance(value, (list, tuple)) else (value,)


def terms(value):
    # Small lexical normalization, not a semantic or attribute classifier.
    def singular(word):
        if word.endswith("ies"):
            return word[:-3] + "y"
        if word.endswith("sses"):
            return word[:-2]
        return word[:-1] if word.endswith("s") and not word.endswith("ss") and len(word) > 3 else word
    return {singular(t) for t in tokenize(_text(value))}


def requirements_from_state(state, *, relax_soft=False):
    hard = state.hard_constraints
    return Requirements(
        category=str(hard.get("category", "")),
        hard_constraints=tuple(str(v) for key, value in hard.items() if key not in {"category", "price_min", "price_max"} for v in sequence(value)),
        soft_preferences=() if relax_soft else tuple(str(v) for prefs in state.soft_preferences.values() for p in prefs for v in sequence(p.value)),
    )


@dataclass(frozen=True)
class RetrievalRequest:
    state: StateSnapshotV2
    strategy: str
    route_weights: dict
    candidate_limit: int = 50
    route_depth: int = 300
    relax_soft: bool = False
    attempt: int = 1

    def __post_init__(self):
        if not isinstance(self.state, StateSnapshotV2):
            raise ValueError("Retrieval requires a versioned full-state snapshot")
        if type(self.candidate_limit) is not int or not 1 <= self.candidate_limit <= 50:
            raise ValueError("candidate_limit must be between 1 and 50")
        if type(self.route_depth) is not int or self.route_depth < 1 or self.attempt not in (1, 2):
            raise ValueError("invalid retrieval depth or retry budget")
        if self.strategy not in {"buying", "browsing", "balanced"}:
            raise ValueError("unsupported strategy")
        if any(not math.isfinite(w) or w < 0 for w in self.route_weights.values()):
            raise ValueError("weights must be finite and nonnegative")

    @classmethod
    def from_state(cls, state):
        buying = state.intent == "buying"
        browsing = state.intent == "browsing"
        return cls(state, "buying" if buying else "browsing" if browsing else "balanced",
                   {"color_normalized_full_and": 3.0 if buying else 1.0,
                    "category_gate": 2.0, "category_or": 2.0 if browsing else 0.5,
                    "evidence": 3.0, "dense": 2.0 if browsing else 1.0})

    def retry(self):
        return replace(self, relax_soft=True, route_depth=self.route_depth * 3, attempt=self.attempt + 1)

    def to_dict(self):
        return {**asdict(self), "state": self.state.to_dict()}


class StateAwareRetriever:
    def __init__(self, catalog_path, *, backend=None, mode="strict"):
        if mode not in {"strict", "recall_compat"}:
            raise ValueError("retrieval mode must be strict or recall_compat")
        self.mode = mode
        self.backend = backend or LiteTop50CandidateGenerator(catalog_path)
        self.products = self.backend.fts.products

    def satisfies(self, asin, state):
        if not state.hard_constraints.get("category"):
            return False
        product = self.products[asin]
        product_terms = terms(" ".join(_text(product.get(f)) for f in PRODUCT_FIELDS))
        category_terms = terms(product.get("categories")) | terms(product.get("title"))
        for name, value in state.hard_constraints.items():
            if name in {"price_min", "price_max"}:
                try:
                    price = float(str(product.get("price", "")).replace("$", "").replace(",", ""))
                except (TypeError, ValueError):
                    return False  # Unknown price cannot certify a hard budget.
                if not math.isfinite(price) or (name == "price_min" and price < float(value)) or (name == "price_max" and price > float(value)):
                    return False
                continue
            # Alternatives within a slot are OR; different slots are AND.
            def matches(v):
                required = terms(v)
                if name == "category":
                    specific = required - {"clothing", "shoe", "jewelry"}
                    return (specific or required) <= category_terms
                if name == "brand":
                    brand_terms = terms(product.get("store")) | terms(product.get("details"))
                    return bool(required) and required <= brand_terms
                if name.startswith("feature_"):
                    # Strip metadata labels with the same module-2 query parser.
                    required = terms(" ".join(parse_text(str(v)).retrieval_terms))
                return bool(required) and required <= product_terms
            if not any(matches(v) for v in sequence(value)):
                return False
        for name, values in state.exclusions.items():
            if any(terms(v) and terms(v) <= product_terms for v in values):
                return False
        return True

    def generate(self, request):
        state = request.state
        requirements = requirements_from_state(state, relax_soft=request.relax_soft)
        if self.mode == "recall_compat":
            legacy = self.backend.generate(requirements, session_id=state.session_id, turn=state.turn)
            return RetrievalResultV2.from_legacy(
                legacy, state_version=state.state_version, state_snapshot=state.to_dict()
            )
        plan = build_retrieval_plan(requirements)
        routes = {name: self.backend.fts.rank(expression, depth=request.route_depth)
                  for name, expression in plan["expressions"].items()}
        routes["evidence"] = self.backend.evidence.rank(requirements, limit=request.route_depth)
        warnings = ["Hard textual constraints use lexical evidence; unknown hard prices are rejected."]
        if hasattr(self.backend, "_dense_ranking"):
            try:
                routes["dense"] = self.backend._dense_ranking(plan["full_query"], depth=min(request.route_depth, len(self.products)))
            except Exception as exc:
                warnings.append(f"Dense failed; lexical routes retained: {type(exc).__name__}")
        else:
            warnings.append("CPU lexical mode: Dense inference is disabled.")
        provenance = {}
        for name, ids in routes.items():
            for rank, asin in enumerate(ids, 1):
                provenance.setdefault(asin, {})[name] = rank
        eligible = [asin for asin in provenance if self.satisfies(asin, state)]
        def score(asin):
            return sum(request.route_weights.get(name, 1.0) / (60 + rank) for name, rank in provenance[asin].items())
        eligible.sort(key=lambda asin: (-score(asin), asin))
        candidates = tuple(Candidate(asin, rank, provenance[asin], {f: self.products[asin].get(f) for f in PRODUCT_FIELDS})
                           for rank, asin in enumerate(eligible[:request.candidate_limit], 1))
        return RetrievalResultV2(
            candidate_set_id=f"{state.session_id}:{state.turn}:v{state.state_version}:a{request.attempt}",
            session_id=state.session_id, turn=state.turn, state_version=state.state_version,
            candidate_limit=request.candidate_limit, candidates=candidates,
            stats=RetrievalStats(len(provenance), len(eligible)),
            state_snapshot=state.to_dict(), legacy_requirements=requirements, warnings=tuple(warnings))

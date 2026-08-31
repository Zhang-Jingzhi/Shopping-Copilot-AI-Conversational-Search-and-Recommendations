"""Run the official public-set evaluator with the rank-after-first agent.

This is a one-off ablation harness. It differs from
``ranking_pipeline.evaluate_agent`` only by importing
``RankAfterFirstAgent``; otherwise the evaluator, catalog, dataset, and output
shape are unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator import local_evaluator

from ranking_pipeline.agent_rank_after_first import RankAfterFirstAgent


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_ROOT = REPOSITORY_ROOT / "retrieval-and-reranking"
RESULTS_ROOT = REPOSITORY_ROOT / "ranking_pipeline" / "results"
DEFAULT_RERANKER_MODEL = (
    REPOSITORY_ROOT
    / "ranking_pipeline"
    / "checkpoints"
    / "qwen3-reranker-0.6B-shopping-lora"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("locked", "hybrid", "local"), default="locked")
    parser.add_argument("--reranker-model", type=Path, default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--retrieval-mode", choices=("lite", "exact"), default="exact")
    parser.add_argument("--catalog", type=Path, default=RETRIEVAL_ROOT / "data" / "catalog.jsonl")
    parser.add_argument("--dataset", type=Path, default=RETRIEVAL_ROOT / "data" / "public_set.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--policy", action="store_true")
    parser.add_argument("--use-state-memory", action="store_true")
    parser.add_argument("--use-intent-router", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    output = (args.output or RESULTS_ROOT / f"{args.mode}-exact-rank-after-first.json").resolve()

    samples = local_evaluator.load_jsonl(args.dataset)
    catalog_ids, categories, products = local_evaluator.catalog_index(args.catalog)
    agent = RankAfterFirstAgent(
        args.catalog,
        retrieval_mode=args.retrieval_mode,
        reranker_mode=args.mode,
        reranker_model=args.reranker_model,
        policy_enabled=args.policy,
        use_state_memory=args.use_state_memory,
        use_intent_router=args.use_intent_router,
    )
    result = local_evaluator.evaluate(
        agent,
        samples,
        catalog_ids,
        categories,
        products,
    )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "sessions"}, indent=2))
    print(
        json.dumps(
            {
                "agent_variant": "rank_after_first",
                "mode": args.mode,
                "retrieval_mode": args.retrieval_mode,
                "hit_rate_at_10": result.get("hit_rate_at_10"),
                "mrr": result.get("mrr"),
                "mttc": result.get("mttc"),
                "technical_score": result.get("recommended_technical_score"),
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

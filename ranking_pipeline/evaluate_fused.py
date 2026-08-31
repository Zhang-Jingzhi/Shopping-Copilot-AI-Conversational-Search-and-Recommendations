"""Evaluate the standalone fused agent against the public 200 set.

This script is deliberately separate from ``evaluate_agent.py`` and does not
modify the existing ranking pipeline.  Use ``--mode fused-locked`` for a
fast deterministic sanity check and ``--mode fused`` for the Qwen pointwise
fusion path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator import local_evaluator

from ranking_pipeline.agent import RankingAgent
from ranking_pipeline.fused_agent import (
    CLEAN_ADAPTER,
    FALLBACK_ADAPTER,
    FusedRankingAgent,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_ROOT = REPOSITORY_ROOT / "retrieval-and-reranking"
RESULTS_ROOT = REPOSITORY_ROOT / "ranking_pipeline" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("fused", "fused-locked", "locked", "local"),
        default="fused-locked",
    )
    parser.add_argument(
        "--reranker-model",
        type=Path,
        default=None,
    )
    parser.add_argument("--retrieval-mode", choices=("lite", "exact"), default="exact")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=RETRIEVAL_ROOT / "data" / "catalog.jsonl",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=RETRIEVAL_ROOT / "data" / "public_set.jsonl",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fuse-retrieval", action="store_true")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=RESULTS_ROOT / "locked-exact.json",
    )
    return parser.parse_args()


def _summary(result: dict) -> dict:
    return {
        "hit_rate_at_10": result.get("hit_rate_at_10"),
        "mrr": result.get("mrr"),
        "mttc": result.get("mttc"),
        "technical_score": result.get("recommended_technical_score"),
        "scenario_metrics": result.get("scenario_metrics"),
    }


def _delta(new: dict, old: dict, key: str) -> float | None:
    try:
        return round(float(new.get(key, 0.0)) - float(old.get(key, 0.0)), 6)
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.reranker_model is None:
        args.reranker_model = (
            CLEAN_ADAPTER if CLEAN_ADAPTER.is_dir() else FALLBACK_ADAPTER
        )

    samples = local_evaluator.load_jsonl(args.dataset)
    catalog_ids, categories, products = local_evaluator.catalog_index(args.catalog)

    if args.mode == "fused":
        agent = FusedRankingAgent(
            args.catalog,
            retrieval_mode=args.retrieval_mode,
            reranker_model=args.reranker_model,
            use_qwen=True,
            use_state_memory=True,
            use_intent_router=True,
            fuse_retrieval=args.fuse_retrieval,
        )
    elif args.mode == "fused-locked":
        agent = FusedRankingAgent(
            args.catalog,
            retrieval_mode=args.retrieval_mode,
            reranker_model=args.reranker_model,
            use_qwen=False,
            use_state_memory=True,
            use_intent_router=True,
            fuse_retrieval=args.fuse_retrieval,
        )
    elif args.mode == "locked":
        agent = RankingAgent(
            args.catalog,
            retrieval_mode=args.retrieval_mode,
            reranker_mode="locked",
            use_state_memory=False,
            use_intent_router=False,
        )
    else:
        agent = RankingAgent(
            args.catalog,
            retrieval_mode=args.retrieval_mode,
            reranker_mode="local",
            reranker_model=args.reranker_model,
            use_state_memory=False,
            use_intent_router=False,
        )

    output = (
        args.output
        or RESULTS_ROOT
        / f"fused-{args.mode}-{args.retrieval_mode}{'-retrieval' if args.fuse_retrieval else ''}.json"
    ).resolve()
    result = local_evaluator.evaluate(
        agent,
        samples,
        catalog_ids,
        categories,
        products,
    )
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(_summary(result), indent=2))

    baseline_data = None
    if args.baseline.exists():
        baseline_data = json.loads(args.baseline.read_text(encoding="utf-8"))

    if baseline_data is not None:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "retrieval_mode": args.retrieval_mode,
                    "fuse_retrieval": args.fuse_retrieval,
                    "output": str(output),
                    "delta_vs_baseline": {
                        "hit_rate_at_10": _delta(
                            result, baseline_data, "hit_rate_at_10"
                        ),
                        "mrr": _delta(result, baseline_data, "mrr"),
                        "mttc": _delta(result, baseline_data, "mttc"),
                        "technical_score": _delta(
                            result, baseline_data, "recommended_technical_score"
                        ),
                    },
                    "intent_override_old": (
                        baseline_data.get("scenario_metrics") or {}
                    ).get("intent_override"),
                    "intent_override_new": (
                        result.get("scenario_metrics") or {}
                    ).get("intent_override"),
                },
                indent=2,
            )
        )
    else:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "output": str(output),
                    "note": f"baseline not found: {args.baseline}",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

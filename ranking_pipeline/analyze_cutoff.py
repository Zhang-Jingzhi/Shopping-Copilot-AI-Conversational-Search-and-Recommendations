"""Offline cutoff analysis from the precomputed Top50 and Top10 JSONL files."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_ROOT = ROOT / "retrieval-and-reranking"
DATA_ROOT = RETRIEVAL_ROOT / "data"
DEFAULT_PUBLIC_TOP50 = DATA_ROOT / "public200_top50.jsonl"
DEFAULT_PUBLIC_TOP10 = DATA_ROOT / "public200_top10.jsonl"


def load_ranking(path: Path) -> dict[str, list[str]]:
    ranking: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ranking[str(row["sample_id"])] = [str(value) for value in row["parent_asins"]]
    return ranking


def load_ground_truth(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            labels[str(row["sample_id"])] = str(row["ground_truth"]["parent_asin"])
    return labels


def reciprocal_rank(rank: int | None) -> float:
    return 0.0 if rank is None else 1.0 / rank


def summarize(
    labels: dict[str, str],
    top50: dict[str, list[str]],
    top10: dict[str, list[str]],
) -> dict[str, object]:
    top50_ranks: list[int] = []
    top20_ranks: list[int] = []
    top10_ranks: list[int] = []
    for sample_id, target in labels.items():
        candidates50 = top50.get(sample_id, [])
        top50_rank = candidates50.index(target) + 1 if target in candidates50 else None
        top20_rank = (
            candidates50.index(target) + 1
            if target in candidates50[:20]
            else None
        )
        candidates10 = top10.get(sample_id, [])
        top10_rank = candidates10.index(target) + 1 if target in candidates10 else None
        if top50_rank is not None:
            top50_ranks.append(top50_rank)
        if top20_rank is not None:
            top20_ranks.append(top20_rank)
        if top10_rank is not None:
            top10_ranks.append(top10_rank)
    total = len(labels)
    return {
        "sample_count": total,
        "top50_coverage": f"{len(top50_ranks)}/{total}",
        "raw_top50_first20_coverage": f"{len(top20_ranks)}/{total}",
        "top10_hit": f"{len(top10_ranks)}/{total}",
        "top20_coverage_lower_bound_from_locked_top10": f"{len(top10_ranks)}/{total}",
        "mrr_at_10": round(
            statistics.mean(reciprocal_rank(rank) for rank in top10_ranks) if top10_ranks else 0.0,
            6,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public-top50",
        type=Path,
        default=DEFAULT_PUBLIC_TOP50,
    )
    parser.add_argument(
        "--public-top10",
        type=Path,
        default=DEFAULT_PUBLIC_TOP10,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATA_ROOT / "public_set.jsonl",
    )
    args = parser.parse_args()
    labels = load_ground_truth(args.dataset)
    top50 = load_ranking(args.public_top50)
    top10 = load_ranking(args.public_top10)
    result = summarize(labels, top50, top10)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

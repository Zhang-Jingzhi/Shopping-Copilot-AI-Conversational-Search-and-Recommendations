"""Generate 3,021 public-schema-compatible sessions with unavailable tags explicit.

This is a contract-matched local development dataset, not a reconstruction of
private sessions.  The public profile tag algorithm and source-event selection
are not released, so tags are deliberately empty and no raw user data is
persisted.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.generate_synthetic_no_profile import (
    DEFAULT_CATALOG,
    DEFAULT_PUBLIC,
    DEFAULT_SEED,
    DEFAULT_TEST,
    DIFFICULTY_BY_SCENARIO,
    allocate_scenario_counts,
    load_catalog,
    load_public_targets,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments" / "results" / "synthetic_contract_matched_all_3021.jsonl"
DEFAULT_QUALITY = ROOT / "experiments" / "results" / "synthetic_contract_matched_all_3021_quality.json"
DEFAULT_TIERS = ROOT / "experiments" / "results" / "product_filter_inference_3021.csv"
DEFAULT_TIER_OUTPUT = ROOT / "experiments" / "results" / "synthetic_contract_matched_all_3021_tiers.jsonl"

SESSION_FIELD_ORDER = (
    "category_bucket",
    "difficulty_bucket",
    "ground_truth",
    "sample_id",
    "scenario_type",
    "user_profile",
)
PROFILE_FIELD_ORDER = (
    "average_prior_rating",
    "preference_tags",
    "purchase_frequency",
    "rating_style",
    "summary",
)
PUBLIC_RATING_STYLE = {
    5: "usually positive",
    4: "mixed",
    3: "critical",
    2: "critical",
    1: "critical",
}
PUBLIC_PURCHASE_FREQUENCY = "3-4 prior purchases"
TAG_POLICY = "empty list: public tag algorithm is not identifiable from released inputs"


class _Edge:
    def __init__(self, target: int, reverse: int, capacity: int) -> None:
        self.target = target
        self.reverse = reverse
        self.capacity = capacity


def _add_edge(graph: list[list[_Edge]], source: int, target: int, capacity: int) -> _Edge:
    forward = _Edge(target, len(graph[target]), capacity)
    reverse = _Edge(source, len(graph[source]), 0)
    graph[source].append(forward)
    graph[target].append(reverse)
    return forward


def _max_flow(graph: list[list[_Edge]], source: int, sink: int) -> int:
    total = 0
    while True:
        level = [-1] * len(graph)
        level[source] = 0
        queue = [source]
        for node in queue:
            for edge in graph[node]:
                if edge.capacity and level[edge.target] < 0:
                    level[edge.target] = level[node] + 1
                    queue.append(edge.target)
        if level[sink] < 0:
            return total
        cursor = [0] * len(graph)

        def send(node: int, remaining: int) -> int:
            if node == sink:
                return remaining
            while cursor[node] < len(graph[node]):
                edge = graph[node][cursor[node]]
                if edge.capacity and level[node] + 1 == level[edge.target]:
                    sent = send(edge.target, min(remaining, edge.capacity))
                    if sent:
                        edge.capacity -= sent
                        graph[edge.target][edge.reverse].capacity += sent
                        return sent
                cursor[node] += 1
            return 0

        while sent := send(source, 10**9):
            total += sent


def _scaled_counts(source_counts: Counter[int], total: int) -> Counter[int]:
    source_total = sum(source_counts.values())
    if not source_total:
        raise ValueError("public rating distribution is empty")
    exact = {rating: total * count / source_total for rating, count in source_counts.items()}
    result = Counter({rating: int(value) for rating, value in exact.items()})
    remaining = total - sum(result.values())
    for rating in sorted(exact, key=lambda item: (-(exact[item] - result[item]), -item))[:remaining]:
        result[rating] += 1
    return result


def load_public_profile_contract(path: Path) -> tuple[Counter[int], set[str], tuple[str, ...]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    field_orders = {tuple(row) for row in rows}
    profile_orders = {tuple(row["user_profile"]) for row in rows}
    frequencies = {str(row["user_profile"]["purchase_frequency"]) for row in rows}
    if field_orders != {SESSION_FIELD_ORDER}:
        raise ValueError(f"unexpected public session schema: {field_orders}")
    if profile_orders != {PROFILE_FIELD_ORDER}:
        raise ValueError(f"unexpected public profile schema: {profile_orders}")
    ratings = Counter(int(row["user_profile"]["average_prior_rating"]) for row in rows)
    return ratings, frequencies, PROFILE_FIELD_ORDER


def scan_target_rating_options(path: Path, catalog_ids: set[str]) -> tuple[dict[str, set[int]], int]:
    options: dict[str, set[int]] = {}
    total_rows = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"parent_asin", "rating"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(f"unexpected test schema: {reader.fieldnames}")
        for row in reader:
            total_rows += 1
            target = str(row["parent_asin"])
            if target in catalog_ids:
                options.setdefault(target, set()).add(int(float(row["rating"])))
    return options, total_rows


def allocate_profile_ratings(
    available_ratings: dict[str, set[int]], desired_counts: Counter[int]
) -> dict[str, int]:
    """Find a deterministic exact assignment of public-style ratings to targets."""
    targets = sorted(available_ratings)
    if sum(desired_counts.values()) != len(targets):
        raise ValueError("desired counts must equal target count")
    ratings = sorted(desired_counts)
    source = 0
    target_offset = 1
    rating_offset = target_offset + len(targets)
    sink = rating_offset + len(ratings)
    graph = [[] for _ in range(sink + 1)]
    references: dict[tuple[str, int], _Edge] = {}
    for index, target in enumerate(targets):
        target_node = target_offset + index
        _add_edge(graph, source, target_node, 1)
        for rating in sorted(available_ratings[target]):
            if rating in desired_counts:
                references[(target, rating)] = _add_edge(
                    graph, target_node, rating_offset + ratings.index(rating), 1
                )
    for index, rating in enumerate(ratings):
        _add_edge(graph, rating_offset + index, sink, desired_counts[rating])
    if _max_flow(graph, source, sink) != len(targets):
        missing = [target for target in targets if not any((target, rating) in references for rating in ratings)]
        raise ValueError(
            f"cannot match the public rating distribution; targets without usable rating: {missing[:10]}"
        )
    assigned: dict[str, int] = {}
    for target in targets:
        selected = [rating for rating in ratings if (target, rating) in references and references[(target, rating)].capacity == 0]
        if len(selected) != 1:
            raise ValueError(f"assignment is not unique for {target}: {selected}")
        assigned[target] = selected[0]
    return assigned


def _profile(rating: int) -> dict[str, Any]:
    style = PUBLIC_RATING_STYLE[rating]
    return {
        "average_prior_rating": float(rating),
        "preference_tags": [],
        "purchase_frequency": PUBLIC_PURCHASE_FREQUENCY,
        "rating_style": style,
        "summary": f"Prior purchases emphasize unavailable; ratings are {style}.",
    }


def build_contract_sessions(
    products: dict[str, dict[str, Any]],
    target_rating_options: dict[str, set[int]],
    desired_rating_counts: Counter[int],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    targets = sorted(target for target in target_rating_options if target in products)
    ratings = allocate_profile_ratings(
        {target: target_rating_options[target] for target in targets}, desired_rating_counts
    )
    rng = random.Random(seed)
    rng.shuffle(targets)
    scenario_counts = allocate_scenario_counts(len(targets))
    scenarios = [scenario for scenario, count in scenario_counts.items() for _ in range(count)]
    rng.shuffle(scenarios)
    return [
        {
            "category_bucket": "clothing",
            "difficulty_bucket": DIFFICULTY_BY_SCENARIO[scenario],
            "ground_truth": {"parent_asin": target},
            "sample_id": f"synthetic_contract_{index:04d}",
            "scenario_type": scenario,
            "user_profile": _profile(ratings[target]),
        }
        for index, (target, scenario) in enumerate(zip(targets, scenarios, strict=True), start=1)
    ]


def validate_contract_sessions(
    sessions: list[dict[str, Any]], catalog_ids: set[str], desired_rating_counts: Counter[int]
) -> None:
    if any(tuple(row) != SESSION_FIELD_ORDER for row in sessions):
        raise ValueError("session fields do not exactly match public schema")
    if any(tuple(row["user_profile"]) != PROFILE_FIELD_ORDER for row in sessions):
        raise ValueError("profile fields do not exactly match public schema")
    targets = [str(row["ground_truth"].get("parent_asin", "")) for row in sessions]
    sample_ids = [str(row["sample_id"]) for row in sessions]
    if len(targets) != len(set(targets)) or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("targets and sample IDs must be unique")
    if any(target not in catalog_ids for target in targets):
        raise ValueError("target outside catalog")
    if any(row["category_bucket"] != "clothing" for row in sessions):
        raise ValueError("category_bucket must match the observed public constant")
    if Counter(int(row["user_profile"]["average_prior_rating"]) for row in sessions) != desired_rating_counts:
        raise ValueError("profile rating distribution does not match calibration")
    if any(row["user_profile"]["purchase_frequency"] != PUBLIC_PURCHASE_FREQUENCY for row in sessions):
        raise ValueError("purchase frequency does not match observed public constant")
    if any(row["user_profile"]["preference_tags"] != [] for row in sessions):
        raise ValueError("preference tags must remain explicitly unavailable")
    for row in sessions:
        profile = row["user_profile"]
        rating = int(profile["average_prior_rating"])
        style = PUBLIC_RATING_STYLE.get(rating)
        if profile["rating_style"] != style:
            raise ValueError("rating style does not match public convention")
        if profile["summary"] != f"Prior purchases emphasize unavailable; ratings are {style}.":
            raise ValueError("summary does not disclose unavailable tags")
        scenario = row["scenario_type"]
        if row["difficulty_bucket"] != DIFFICULTY_BY_SCENARIO[scenario]:
            raise ValueError("difficulty bucket does not match scenario")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_tiers(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row["parent_asin"]): {
                "quality_tier": str(row["quality_tier"]),
                "selection_frequency": float(row["selection_frequency"]),
                "predicted_pool_member": str(row["predicted_pool_member"]).lower() == "true",
            }
            for row in reader
        }


def generate(
    catalog_path: Path,
    public_path: Path,
    test_path: Path,
    tier_path: Path,
    output_path: Path,
    quality_path: Path,
    tier_output_path: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    from evaluator.local_evaluator import materialize_hidden_fields

    public_ratings, public_frequencies, _ = load_public_profile_contract(public_path)
    if public_frequencies != {PUBLIC_PURCHASE_FREQUENCY}:
        raise ValueError(f"unexpected public purchase-frequency values: {public_frequencies}")
    catalog = load_catalog(catalog_path)
    public_targets = load_public_targets(public_path)
    rating_options, test_rows = scan_target_rating_options(test_path, set(catalog))
    rating_options = {target: values for target, values in rating_options.items() if target not in public_targets}
    desired_ratings = _scaled_counts(public_ratings, len(rating_options))
    sessions = build_contract_sessions(catalog, rating_options, desired_ratings, seed=seed)
    validate_contract_sessions(sessions, set(catalog), desired_ratings)

    materialized = 0
    for sample in sessions:
        card, behavior = materialize_hidden_fields(sample, catalog)
        if not card.get("hard_constraints") or behavior.get("scenario_type") != sample["scenario_type"]:
            raise ValueError(f"evaluator failed to materialize {sample['sample_id']}")
        materialized += 1

    tiers = _load_tiers(tier_path)
    tier_rows = []
    for sample in sessions:
        target = str(sample["ground_truth"]["parent_asin"])
        if target not in tiers:
            raise ValueError(f"missing tier for {target}")
        tier_rows.append({"sample_id": sample["sample_id"], "target_parent_asin": target, **tiers[target]})

    _write_jsonl(output_path, sessions)
    _write_jsonl(tier_output_path, tier_rows)
    quality = {
        "dataset_kind": "strict public-schema-compatible local proxy; not organizer/private data",
        "generator": "experiments.generate_contract_matched_sessions",
        "generator_version": 1,
        "seed": seed,
        "counts": {
            "catalog_products": len(catalog),
            "test_rows_scanned": test_rows,
            "public_targets_excluded": len(public_targets),
            "generated_sessions": len(sessions),
            "official_hidden_fields_materialized": materialized,
        },
        "schema": {
            "session_fields_exactly_match_public": list(SESSION_FIELD_ORDER),
            "profile_fields_exactly_match_public": list(PROFILE_FIELD_ORDER),
            "category_bucket_constant": "clothing",
        },
        "profile_calibration": {
            "public_rating_counts": dict(sorted(public_ratings.items())),
            "generated_rating_counts": dict(sorted(desired_ratings.items())),
            "purchase_frequency": PUBLIC_PURCHASE_FREQUENCY,
            "rating_style_mapping": PUBLIC_RATING_STYLE,
            "tag_policy": TAG_POLICY,
            "tag_count_per_session": 0,
        },
        "scenario_counts": dict(sorted(Counter(row["scenario_type"] for row in sessions).items())),
        "tier_counts": dict(sorted(Counter(row["quality_tier"] for row in tier_rows).items())),
        "input_sha256": {
            "catalog": _sha256(catalog_path),
            "public_set": _sha256(public_path),
            "test_split": _sha256(test_path),
            "product_tiers": _sha256(tier_path),
        },
        "outputs": {
            "sessions": {"path": str(output_path.relative_to(ROOT)), "sha256": _sha256(output_path)},
            "tiers": {"path": str(tier_output_path.relative_to(ROOT)), "sha256": _sha256(tier_output_path)},
        },
        "confirmed": [
            "Every session has exactly the public session and profile field names in the same order.",
            "category_bucket, scenario proportions, rating-style mapping, and purchase-frequency label follow public observations.",
            "Each assigned average_prior_rating proxy is an observed target-interaction rating available for that target in the local LLO test split; this does not establish a user-history average.",
            "The unmodified local evaluator materializes every session.",
        ],
        "not_reconstructed": [
            "Organizer source-user selection for multi-event targets.",
            "Organizer purchase-frequency, average-rating, and preference-tag algorithms.",
            "The source user's prior rating sequence and the aggregation that would produce a real average_prior_rating.",
            "Preference tags; they are deliberately [] and the summary says unavailable.",
            "Private 800 target membership or profile distribution.",
        ],
    }
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--tiers", type=Path, default=DEFAULT_TIERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quality", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--tier-output", type=Path, default=DEFAULT_TIER_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    print(json.dumps(generate(
        args.catalog, args.public, args.test, args.tiers, args.output, args.quality, args.tier_output, seed=args.seed
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

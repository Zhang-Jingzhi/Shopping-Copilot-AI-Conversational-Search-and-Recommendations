"""Infer plausible product/interaction filters for the organizer candidate pool."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments.analyze_product_selection_position import midrank_percentiles


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data" / "catalog.jsonl"
DEFAULT_PUBLIC = ROOT / "data" / "public_set.jsonl"
DEFAULT_TEST = ROOT / "data" / "upstream" / "amazon_reviews_2023" / "5core_llo" / "Clothing_Shoes_and_Jewelry.test.csv.gz"
DEFAULT_PRODUCTS = ROOT / "experiments" / "results" / "product_selection_position_3221.csv"
DEFAULT_OUTPUT = ROOT / "experiments" / "results" / "product_filter_inference_audit.json"
DEFAULT_CSV = ROOT / "experiments" / "results" / "product_filter_inference_3021.csv"
DEFAULT_RESULTS = ROOT / "experiments" / "product_filter_inference" / "results.tsv"
TARGET_ROWS = 10_187
TARGET_PRODUCTS = 1_406
PUBLIC_PRODUCTS = 200
NONPUBLIC_POOL_PRODUCTS = TARGET_PRODUCTS - PUBLIC_PRODUCTS


def select_capped_top_rows(
    events: list[dict[str, Any]],
    score_field: str,
    *,
    cap_per_product: int,
    target_rows: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        events,
        key=lambda row: (-float(row[score_field]), str(row.get("tie", ""))),
    )
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for row in ordered:
        target = str(row["target"])
        if counts[target] >= cap_per_product:
            continue
        counts[target] += 1
        selected.append(row)
        if len(selected) == target_rows:
            break
    return selected


def pareto_frontier(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("row_error", "product_error", "public_misses")
    frontier = []
    for candidate in rules:
        dominated = any(
            all(float(other[field]) <= float(candidate[field]) for field in fields)
            and any(float(other[field]) < float(candidate[field]) for field in fields)
            for other in rules
            if other is not candidate
        )
        if not dominated:
            frontier.append(candidate)
    return frontier


def assign_consensus_tiers(
    rows: list[dict[str, Any]], *, predicted_pool_size: int
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (-float(row["selection_frequency"]), str(row["parent_asin"])),
    )
    predicted_ids = {row["parent_asin"] for row in ordered[:predicted_pool_size]}
    output = []
    for source in rows:
        row = dict(source)
        frequency = float(row["selection_frequency"])
        if frequency >= 0.8:
            tier = "high_confidence"
        elif frequency >= 0.5:
            tier = "probable"
        elif frequency >= 0.2:
            tier = "uncertain"
        else:
            tier = "low_likelihood"
        row["quality_tier"] = tier
        row["predicted_pool_member"] = row["parent_asin"] in predicted_ids
        output.append(row)
    return output


def _bool(value: Any) -> bool:
    return str(value).lower() == "true"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_product_rows(path: Path) -> list[dict[str, Any]]:
    numeric_fields = {
        "rating_number", "test_event_count", "average_rating", "metadata_nonempty_count",
        "feature_count", "description_char_count", "details_count",
    }
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["is_public"] = _bool(row["is_public"])
        for field in numeric_fields:
            row[field] = float(row[field] or 0)
        for field, value in list(row.items()):
            if field.endswith("_percentile") and value != "":
                row[field] = float(value)
    return rows


def load_events(
    test_path: Path, product_ids: set[str], catalog_ids: set[str]
) -> tuple[list[dict[str, Any]], dict[str, Counter[tuple[int, int, int]]]]:
    events: list[dict[str, Any]] = []
    signatures: dict[str, Counter[tuple[int, int, int]]] = defaultdict(Counter)
    with gzip.open(test_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            target = str(row["parent_asin"])
            if target not in product_ids:
                continue
            history = [value for value in row["history"].split() if value]
            history_catalog_count = sum(value in catalog_ids for value in history)
            rating = int(float(row["rating"]))
            event = {
                "target": target,
                "rating": rating,
                "history_length": len(history),
                "history_catalog_count": history_catalog_count,
                "history_catalog_ratio": history_catalog_count / len(history) if history else 0.0,
                "tie": f"{row['timestamp']}|{row['user_id']}|{target}",
            }
            events.append(event)
            signatures[target][(rating, len(history), history_catalog_count)] += 1
    return events, signatures


def rule_metrics(
    name: str,
    rows: int,
    products: set[str],
    public_ids: set[str],
    **parameters: Any,
) -> dict[str, Any]:
    public_hits = len(products & public_ids)
    row_error = abs(rows - TARGET_ROWS) / TARGET_ROWS
    product_error = abs(len(products) - TARGET_PRODUCTS) / TARGET_PRODUCTS
    public_misses = PUBLIC_PRODUCTS - public_hits
    return {
        "name": name,
        "rows": rows,
        "products": len(products),
        "public_hits": public_hits,
        "public_misses": public_misses,
        "row_error": row_error,
        "product_error": product_error,
        "objective": row_error + product_error + 5 * public_misses / PUBLIC_PRODUCTS,
        "parameters": parameters,
    }


def run_simple_threshold_search(
    signatures: dict[str, Counter[tuple[int, int, int]]],
    products_by_id: dict[str, dict[str, Any]],
    public_ids: set[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for minimum_rating in range(1, 6):
        for minimum_history_length in range(4, 31):
            for minimum_history_catalog_count in range(0, 4):
                qualified = {
                    target: sum(
                        count
                        for (rating, history_length, history_catalog_count), count in target_signatures.items()
                        if rating >= minimum_rating
                        and history_length >= minimum_history_length
                        and history_catalog_count >= minimum_history_catalog_count
                    )
                    for target, target_signatures in signatures.items()
                }
                for minimum_average_rating in (0.0, 3.5, 4.0):
                    for minimum_metadata_nonempty in (0, 7, 8):
                        eligible_counts = {
                            target: count
                            for target, count in qualified.items()
                            if count > 0
                            and float(products_by_id[target]["average_rating"]) >= minimum_average_rating
                            and float(products_by_id[target]["metadata_nonempty_count"]) >= minimum_metadata_nonempty
                        }
                        product_ids = set(eligible_counts)
                        for cap in (None, *range(4, 51)):
                            rows = sum(
                                count if cap is None else min(count, cap)
                                for count in eligible_counts.values()
                            )
                            results.append(
                                rule_metrics(
                                    "simple_threshold",
                                    rows,
                                    product_ids,
                                    public_ids,
                                    minimum_rating=minimum_rating,
                                    minimum_history_length=minimum_history_length,
                                    minimum_history_catalog_count=minimum_history_catalog_count,
                                    minimum_average_rating=minimum_average_rating,
                                    minimum_metadata_nonempty=minimum_metadata_nonempty,
                                    cap_per_product=cap,
                                )
                            )
    return results


def _ranked_score(event: dict[str, Any], product: dict[str, Any], variant: str) -> float:
    history = event["history_length"] / 30
    rating = (event["rating"] - 1) / 4
    catalog_count = min(event["history_catalog_count"], 5) / 5
    catalog_ratio = event["history_catalog_ratio"]
    event_popularity = float(product["test_event_count_family_vs_nonpublic_percentile"])
    rating_popularity = float(product["rating_number_family_vs_nonpublic_percentile"])
    metadata = float(product["metadata_nonempty_count"]) / 9
    scores = {
        "history_only": history,
        "rating_only": rating,
        "catalog_history_only": catalog_count,
        "user_quality": 0.60 * history + 0.25 * catalog_count + 0.15 * rating,
        "user_quality_ratio": 0.35 * history + 0.35 * catalog_ratio + 0.30 * rating,
        "product_popularity": 0.55 * event_popularity + 0.45 * rating_popularity,
        "user_plus_product": 0.30 * history + 0.15 * catalog_count + 0.10 * rating + 0.30 * event_popularity + 0.15 * rating_popularity,
        "interaction_first": 0.30 * history + 0.10 * rating + 0.40 * event_popularity + 0.20 * rating_popularity,
        "metadata_assisted": 0.25 * history + 0.10 * rating + 0.30 * event_popularity + 0.20 * rating_popularity + 0.15 * metadata,
    }
    return scores[variant]


def run_ranked_cap_search(
    events: list[dict[str, Any]],
    products_by_id: dict[str, dict[str, Any]],
    public_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    results: list[dict[str, Any]] = []
    selected_sets: dict[str, set[str]] = {}
    variants = (
        "history_only", "rating_only", "catalog_history_only", "user_quality",
        "user_quality_ratio", "product_popularity", "user_plus_product",
        "interaction_first", "metadata_assisted",
    )
    for variant in variants:
        scored = [
            {**event, "score": _ranked_score(event, products_by_id[event["target"]], variant)}
            for event in events
        ]
        for cap in range(4, 21):
            selected = select_capped_top_rows(
                scored, "score", cap_per_product=cap, target_rows=TARGET_ROWS
            )
            product_ids = {row["target"] for row in selected}
            key = f"ranked:{variant}:cap={cap}"
            results.append(
                rule_metrics(
                    "ranked_capped_rows",
                    len(selected),
                    product_ids,
                    public_ids,
                    variant=variant,
                    cap_per_product=cap,
                )
            )
            selected_sets[key] = product_ids
    return results, selected_sets


def _add_signature_features(
    products: list[dict[str, Any]], signatures: dict[str, Counter[tuple[int, int, int]]]
) -> None:
    definitions = {
        "quality_history_13": lambda rating, history, catalog: history >= 13,
        "quality_history_10_rating_4": lambda rating, history, catalog: history >= 10 and rating >= 4,
        "quality_catalog_history": lambda rating, history, catalog: catalog >= 1,
    }
    for name, predicate in definitions.items():
        values = []
        for product in products:
            value = sum(
                count
                for (rating, history, catalog), count in signatures[product["parent_asin"]].items()
                if predicate(rating, history, catalog)
            )
            product[name] = value
            values.append(value)
        for product, percentile in zip(products, midrank_percentiles(values)):
            product[f"{name}_percentile"] = percentile
    max_histories = [
        max((history for (_, history, _), count in signatures[row["parent_asin"]].items() if count), default=0)
        for row in products
    ]
    for product, value, percentile in zip(products, max_histories, midrank_percentiles(max_histories)):
        product["max_history_length"] = value
        product["max_history_percentile"] = percentile


def run_product_score_search(
    products: list[dict[str, Any]], public_ids: set[str]
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    results: list[dict[str, Any]] = []
    selected_sets: dict[str, set[str]] = {}
    for event_weight in range(0, 5):
        for rating_weight in range(0, 4):
            for history_weight in range(0, 3):
                for quality_weight in range(0, 4):
                    for metadata_weight in (0.0, 0.5):
                        weights = (event_weight, rating_weight, history_weight, quality_weight, metadata_weight)
                        if sum(weights) == 0:
                            continue
                        scored = []
                        for product in products:
                            score = (
                                event_weight * float(product["test_event_count_family_vs_nonpublic_percentile"])
                                + rating_weight * float(product["rating_number_family_vs_nonpublic_percentile"])
                                + history_weight * float(product["max_history_percentile"])
                                + quality_weight * float(product["quality_history_13_percentile"])
                                + metadata_weight * float(product["metadata_nonempty_count_stable_peer_group_vs_nonpublic_percentile"])
                            ) / sum(weights)
                            scored.append((score, product["parent_asin"], product))
                        selected = sorted(scored, key=lambda item: (-item[0], item[1]))[:TARGET_PRODUCTS]
                        product_ids = {item[1] for item in selected}
                        best_cap = min(
                            range(4, 21),
                            key=lambda cap: abs(
                                sum(min(int(item[2]["test_event_count"]), cap) for item in selected)
                                - TARGET_ROWS
                            ),
                        )
                        rows = sum(min(int(item[2]["test_event_count"]), best_cap) for item in selected)
                        key = "product_score:" + ":".join(map(str, weights))
                        results.append(
                            rule_metrics(
                                "weighted_product_score",
                                rows,
                                product_ids,
                                public_ids,
                                event_weight=event_weight,
                                rating_weight=rating_weight,
                                history_weight=history_weight,
                                quality_weight=quality_weight,
                                metadata_weight=metadata_weight,
                                cap_per_product=best_cap,
                            )
                        )
                        selected_sets[key] = product_ids
    return results, selected_sets


def quantile_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    def at(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]
    return {
        "minimum": ordered[0], "p10": at(0.10), "p25": at(0.25),
        "median": statistics.median(ordered), "mean": statistics.fmean(ordered),
        "p75": at(0.75), "p90": at(0.90), "maximum": ordered[-1],
    }


def build_consensus(
    products: list[dict[str, Any]],
    model_results: list[dict[str, Any]],
    selected_sets: dict[str, set[str]],
    public_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    product_result_keys = {
        "product_score:" + ":".join(map(str, (
            row["parameters"]["event_weight"], row["parameters"]["rating_weight"],
            row["parameters"]["history_weight"], row["parameters"]["quality_weight"],
            row["parameters"]["metadata_weight"],
        ))): row
        for row in model_results
        if row["name"] == "weighted_product_score"
    }
    ranked_models = sorted(
        product_result_keys.items(),
        key=lambda item: (item[1]["public_misses"], item[1]["row_error"], item[1]["objective"]),
    )
    best_public_misses = ranked_models[0][1]["public_misses"]
    retained = [
        (key, result) for key, result in ranked_models
        if result["public_misses"] <= best_public_misses + 3
    ][:100]
    frequencies = Counter()
    for key, _ in retained:
        for target in selected_sets[key]:
            if target not in public_ids:
                frequencies[target] += 1
    nonpublic = [row for row in products if row["parent_asin"] not in public_ids]
    consensus_rows = [
        {
            **row,
            "selection_frequency": frequencies[row["parent_asin"]] / len(retained),
        }
        for row in nonpublic
    ]
    return assign_consensus_tiers(
        consensus_rows, predicted_pool_size=NONPUBLIC_POOL_PRODUCTS
    ), [result for _, result in retained]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--public", type=Path, default=DEFAULT_PUBLIC)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    products = load_product_rows(args.products)
    products_by_id = {row["parent_asin"]: row for row in products}
    public_ids = {row["parent_asin"] for row in products if row["is_public"]}
    catalog_ids = {row["parent_asin"] for row in load_jsonl(args.catalog)}
    events, signatures = load_events(args.test, set(products_by_id), catalog_ids)
    _add_signature_features(products, signatures)

    simple_results = run_simple_threshold_search(signatures, products_by_id, public_ids)
    ranked_results, ranked_sets = run_ranked_cap_search(events, products_by_id, public_ids)
    product_results, product_sets = run_product_score_search(products, public_ids)
    all_results = simple_results + ranked_results + product_results
    consensus, retained_models = build_consensus(
        products, product_results, product_sets, public_ids
    )

    best_by_family = {
        name: min((row for row in all_results if row["name"] == name), key=lambda row: row["objective"])
        for name in {row["name"] for row in all_results}
    }
    best_count_fit_by_family = {
        name: min(
            (row for row in all_results if row["name"] == name),
            key=lambda row: (row["row_error"] + row["product_error"], row["public_misses"]),
        )
        for name in {row["name"] for row in all_results}
    }
    best_public_complete_by_family = {
        name: min(
            (row for row in all_results if row["name"] == name and row["public_misses"] == 0),
            key=lambda row: row["row_error"] + row["product_error"],
            default=None,
        )
        for name in {row["name"] for row in all_results}
    }
    near_count_rules = [
        row for row in all_results
        if row["row_error"] <= 0.05 and row["product_error"] <= 0.05
    ]
    public_products = [row for row in products if row["is_public"]]
    summary_fields = (
        "rating_number", "test_event_count", "max_history_length",
        "quality_history_13", "quality_history_10_rating_4", "quality_catalog_history",
    )
    tier_counts = Counter(row["quality_tier"] for row in consensus)
    predicted = [row for row in consensus if row["predicted_pool_member"]]
    audit = {
        "status": "non_identifiable_proxy_with_consensus_stratification",
        "targets": {"quality_rows": TARGET_ROWS, "quality_products": TARGET_PRODUCTS, "public_products": PUBLIC_PRODUCTS},
        "observed": {"events": len(events), "products": len(products), "public_products": len(public_ids)},
        "public_raw_signal_ranges": {
            field: quantile_summary([float(row[field]) for row in public_products])
            for field in summary_fields
        },
        "experiment_counts": {
            "simple_threshold_rules": len(simple_results),
            "ranked_cap_rules": len(ranked_results),
            "weighted_product_rules": len(product_results),
            "total_rules": len(all_results),
        },
        "best_by_family": best_by_family,
        "best_count_fit_by_family": best_count_fit_by_family,
        "best_public_complete_by_family": best_public_complete_by_family,
        "near_count_rule_summary": {
            "rule_count": len(near_count_rules),
            "maximum_public_hits": max((row["public_hits"] for row in near_count_rules), default=0),
            "best_public_coverage_rule": min(
                near_count_rules,
                key=lambda row: (row["public_misses"], row["row_error"] + row["product_error"]),
                default=None,
            ),
        },
        "pareto_frontier": sorted(
            pareto_frontier(all_results), key=lambda row: (row["public_misses"], row["row_error"] + row["product_error"])
        )[:100],
        "exact_count_matches": [
            row for row in all_results
            if row["rows"] == TARGET_ROWS and row["products"] == TARGET_PRODUCTS
        ][:100],
        "consensus": {
            "retained_model_count": len(retained_models),
            "best_model_public_misses": min(row["public_misses"] for row in retained_models),
            "tier_counts": dict(sorted(tier_counts.items())),
            "predicted_nonpublic_pool_count": len(predicted),
            "predicted_full_pool_count_with_public": len(predicted) + len(public_ids),
            "selection_frequency_summary": quantile_summary(
                [float(row["selection_frequency"]) for row in consensus]
            ),
        },
        "limitations": [
            "Counts alone do not identify the organizer's 1,406 product IDs.",
            "Public products are observed positives, while the 3,021 comparison products are positive-unlabeled.",
            "Model retention uses public coverage and therefore estimates public-like selection, not private labels.",
            "Raw review text, verified_purchase, helpful_vote, and organizer generation-quality labels remain unavailable.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_fields = [
        "parent_asin", "title", "family", "leaf_category", "quality_tier",
        "predicted_pool_member", "selection_frequency", "rating_number", "test_event_count",
        "max_history_length", "quality_history_13", "quality_history_10_rating_4",
        "quality_catalog_history", "average_rating", "metadata_nonempty_count",
    ]
    with args.csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            sorted(consensus, key=lambda row: (-row["selection_frequency"], row["parent_asin"]))
        )
    args.results.parent.mkdir(parents=True, exist_ok=True)
    with args.results.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("family\tobjective\trows\tproducts\tpublic_hits\tparameters\n")
        for name, row in sorted(best_by_family.items()):
            handle.write(
                f"{name}\t{row['objective']:.8f}\t{row['rows']}\t{row['products']}\t{row['public_hits']}\t"
                f"{json.dumps(row['parameters'], ensure_ascii=False, sort_keys=True)}\n"
            )
    print(json.dumps({
        "experiment_counts": audit["experiment_counts"],
        "best_by_family": best_by_family,
        "best_count_fit_by_family": best_count_fit_by_family,
        "best_public_complete_by_family": best_public_complete_by_family,
        "near_count_rule_summary": audit["near_count_rule_summary"],
        "exact_count_matches": len(audit["exact_count_matches"]),
        "consensus": audit["consensus"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

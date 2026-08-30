"""Deterministic public-set training data for the Qwen3 reranker.

The evaluator materializes hidden requirements from product metadata. For local
reranker training we reproduce that public logic from the participant kit, then
join the already-materialized Top50 candidate list with the frozen catalog.
The organizer's private 800 sessions are never used and no private label is
reconstructed.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from techjam_agent.contracts import Requirements
from ranking_pipeline.qwen_reranker import format_pair, product_text, query_text


MATERIALS = (
    "cotton",
    "polyester",
    "nylon",
    "leather",
    "wool",
    "spandex",
    "silk",
    "rayon",
    "fabric",
)
COLORS = (
    "black",
    "white",
    "blue",
    "red",
    "pink",
    "green",
    "brown",
    "gray",
    "grey",
    "purple",
    "yellow",
    "orange",
)
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")
MATERIAL_RE = re.compile(rf"\b({'|'.join(MATERIALS)})\b", re.I)
COLOR_RE = re.compile(rf"\b({'|'.join(COLORS)})\b", re.I)


def _flatten_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field in SEARCH_FIELDS:
        value = product.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def clean_constraint(value: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def intent_card(product: dict, limit: int = 180) -> dict:
    title = clean_constraint(str(product.get("title") or "product"), limit)
    candidates = [
        *_flatten_values(product.get("features")),
        *_flatten_values(product.get("details")),
    ]
    corpus = searchable_text(product)
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    if material:
        candidates.insert(0, material.group(1).lower())
    if color:
        candidates.insert(1, f"color: {color.group(1).lower()}")
    if product.get("price") not in (None, ""):
        candidates.append(f"budget around ${product['price']}")
    cleaned = list(
        dict.fromkeys(clean_constraint(item, limit) for item in candidates if clean_constraint(item, limit))
    )
    if not cleaned:
        cleaned = [title]
    return {
        "target_category": title,
        "hard_constraints": cleaned[:2],
        "soft_preferences": cleaned[2:4] or cleaned[:1],
    }


def requirements_from_product(product: dict) -> Requirements:
    card = intent_card(product)
    return Requirements(
        category=card["target_category"],
        hard_constraints=tuple(card["hard_constraints"]),
        soft_preferences=tuple(card["soft_preferences"]),
    )


@dataclass(frozen=True)
class RerankTrainingExample:
    query: str
    document: str
    label: float
    parent_asin: str


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_catalog(path: str | Path) -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            catalog[str(product["parent_asin"])] = product
    return catalog


def load_top50(path: str | Path) -> dict[str, list[str]]:
    top50: dict[str, list[str]] = {}
    for row in load_jsonl(path):
        top50[str(row["sample_id"])] = [str(value) for value in row["parent_asins"]]
    return top50


def build_public_training_examples(
    public_set_path: str | Path,
    public_top50_path: str | Path,
    catalog_path: str | Path,
    *,
    negatives_per_positive: int = 4,
    seed: int = 0,
    limit: int | None = None,
) -> list[RerankTrainingExample]:
    samples = load_jsonl(public_set_path)
    top50 = load_top50(public_top50_path)
    catalog = load_catalog(catalog_path)
    rng = random.Random(seed)
    examples: list[RerankTrainingExample] = []
    for sample in samples:
        sample_id = str(sample["sample_id"])
        target = str(sample["ground_truth"]["parent_asin"])
        candidates = top50.get(sample_id, [])
        if target not in candidates:
            continue
        target_product = catalog.get(target)
        if target_product is None:
            continue
        query = query_text(requirements_from_product(target_product))
        examples.append(
            RerankTrainingExample(
                query=query,
                document=product_text(_candidate_product(target, target_product)),
                label=1.0,
                parent_asin=target,
            )
        )
        negatives = [item for item in candidates if item != target]
        rng.shuffle(negatives)
        for negative_id in negatives[:negatives_per_positive]:
            negative_product = catalog.get(negative_id)
            if negative_product is None:
                continue
            examples.append(
                RerankTrainingExample(
                    query=query,
                    document=product_text(_candidate_product(negative_id, negative_product)),
                    label=0.0,
                    parent_asin=negative_id,
                )
            )
        if limit is not None and len(examples) >= limit:
            break
    return examples


def _candidate_product(parent_asin: str, product: dict):
    """Wrap a catalog row into the minimal Candidate-like shape expected by helpers."""

    from techjam_agent.contracts import Candidate

    public_product = {
        field: product.get(field)
        for field in ("title", "categories", "features", "details", "description", "store")
    }
    return Candidate(
        parent_asin=parent_asin,
        candidate_rank=1,
        source_ranks={"train": 1},
        product=public_product,
    )


def examples_to_pairs(examples: Iterable[RerankTrainingExample]) -> list[tuple[str, str]]:
    return [(example.query, example.document) for example in examples]

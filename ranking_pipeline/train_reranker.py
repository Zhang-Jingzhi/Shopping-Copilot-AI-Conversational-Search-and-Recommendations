"""Train a lightweight Qwen3-Reranker scoring head with aligned data.

The official rules allow local scoring logic and prompt tuning, while
full-parameter training of base foundational LLMs is out of scope. This script
therefore freezes the transformer trunk and trains only the final
classification/score parameters. By default it combines the public 200 gold
sessions with the synthetic 3,021 proxy sessions for distribution alignment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from ranking_pipeline.qwen_reranker import DEFAULT_MODEL, format_pair
from ranking_pipeline.training_data import (
    RerankTrainingExample,
    build_public_training_examples,
    build_synthetic_training_examples,
)
from ranking_pipeline.distribution_alignment import (
    build_aligned_training_examples,
    summarize_examples,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_ROOT = REPOSITORY_ROOT / "retrieval-and-reranking"
DATA_ROOT = RETRIEVAL_ROOT / "data"
DEFAULT_PUBLIC_SET = DATA_ROOT / "public_set.jsonl"
DEFAULT_CATALOG = DATA_ROOT / "catalog.jsonl"
DEFAULT_SYNTHETIC_SET = (
    REPOSITORY_ROOT / "synthetic-data-3021" / "data" / "synthetic_contract_matched_all_3021.jsonl"
)
DEFAULT_SYNTHETIC_TIERS = (
    REPOSITORY_ROOT / "synthetic-data-3021" / "data" / "synthetic_contract_matched_all_3021_tiers.jsonl"
)
DEFAULT_SYNTHETIC_PRODUCTS = (
    REPOSITORY_ROOT / "synthetic-data-3021" / "data" / "product_filter_inference_3021.csv"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "ranking_pipeline" / "checkpoints" / "qwen3-reranker-0.6b-shopping"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-strategy",
        choices=("public", "synthetic", "aligned"),
        default="aligned",
    )
    parser.add_argument("--public-set", type=Path, default=DEFAULT_PUBLIC_SET)
    parser.add_argument("--public-top50", type=Path, default=None)
    parser.add_argument("--synthetic-set", type=Path, default=DEFAULT_SYNTHETIC_SET)
    parser.add_argument("--synthetic-tiers", type=Path, default=DEFAULT_SYNTHETIC_TIERS)
    parser.add_argument("--synthetic-products", type=Path, default=DEFAULT_SYNTHETIC_PRODUCTS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--negatives-per-positive", type=int, default=4, help="Public negatives per positive")
    parser.add_argument("--synthetic-negatives-per-positive", type=int, default=4)
    parser.add_argument("--public-positive-weight", type=float, default=5.0)
    parser.add_argument("--public-negative-weight", type=float, default=1.0)
    parser.add_argument("--synthetic-tier-filter", nargs="+", default=("high_confidence", "probable"))
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _split_examples(examples: list[RerankTrainingExample], fraction: float, seed: int):
    # Keep one positive and its associated negatives together in the same split.
    positive_groups: dict[str, list[RerankTrainingExample]] = {}
    order: list[str] = []
    last_group: str | None = None
    for example in examples:
        if example.label == 1.0:
            group_id = str(len(order))
            order.append(group_id)
            positive_groups[group_id] = [example]
            last_group = group_id
        else:
            if last_group is None:
                continue
            if last_group not in positive_groups:
                positive_groups[last_group] = []
            positive_groups[last_group].append(example)
    random.Random(seed).shuffle(order)
    eval_count = max(1, int(len(order) * fraction))
    eval_groups = set(order[:eval_count])
    train: list[RerankTrainingExample] = []
    valid: list[RerankTrainingExample] = []
    for group_id in order:
        target = valid if group_id in eval_groups else train
        target.extend(positive_groups[group_id])
    return train, valid


def _classifier_parameter(name: str) -> bool:
    lowered = name.lower()
    return any(
        marker in lowered
        for marker in ("classifier", "score", "classifier_out", "pooler")
    )


def _load_model(model_name: str, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Use fp32 for the trainable classifier so gradients stay stable without
    # mixed-precision master weights. The 0.6B trunk still fits comfortably in
    # a 6GB consumer GPU for small batch sizes.
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    trainable_count = 0
    for name, parameter in model.named_parameters():
        requires_grad = _classifier_parameter(name)
        parameter.requires_grad = requires_grad
        trainable_count += int(requires_grad)
    if trainable_count == 0:
        # Fall back to all parameters if this conversion exposes no conventional
        # classifier name. This is a safeguard, not the intended path.
        for parameter in model.parameters():
            parameter.requires_grad = True
        trainable_count = sum(parameter.numel() for parameter in model.parameters())
    model.train()
    return tokenizer, model, trainable_count


def _encode_batch(tokenizer, examples: list[RerankTrainingExample], device: str, max_length: int):
    texts = [format_pair(example.query, example.document) for example in examples]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation="longest_first",
        max_length=max_length,
    )
    labels = [example.label for example in examples]
    weights = [example.weight for example in examples]
    return (
        {name: value.to(device) for name, value in inputs.items()},
        labels,
        weights,
    )


def train(args: argparse.Namespace) -> None:
    import torch
    from torch import nn

    if args.epochs is None:
        args.epochs = 1 if args.data_strategy == "public" else 2

    if args.data_strategy == "public":
        examples = build_public_training_examples(
            args.public_set,
            args.public_top50,
            args.catalog,
            negatives_per_positive=args.negatives_per_positive,
            negative_pool_csv_path=args.synthetic_products,
            seed=args.seed,
            limit=args.limit,
            positive_weight=args.public_positive_weight,
            negative_weight=args.public_negative_weight,
        )
    elif args.data_strategy == "synthetic":
        examples = build_synthetic_training_examples(
            args.synthetic_set,
            args.catalog,
            product_csv_path=args.synthetic_products,
            tiers_path=args.synthetic_tiers,
            negatives_per_positive=args.synthetic_negatives_per_positive,
            seed=args.seed,
            limit=args.limit,
            tier_filter=tuple(args.synthetic_tier_filter),
        )
    else:
        examples = build_aligned_training_examples(
            args.public_set,
            args.public_top50,
            args.synthetic_set,
            args.catalog,
            synthetic_product_csv_path=args.synthetic_products,
            synthetic_tiers_path=args.synthetic_tiers,
            public_negatives_per_positive=args.negatives_per_positive,
            synthetic_negatives_per_positive=args.synthetic_negatives_per_positive,
            public_positive_weight=args.public_positive_weight,
            public_negative_weight=args.public_negative_weight,
            synthetic_tier_filter=tuple(args.synthetic_tier_filter),
            seed=args.seed,
            limit=args.limit,
        )
    if not examples:
        raise RuntimeError("No training examples could be built")
    public_validation_examples: list[RerankTrainingExample] = []
    if args.data_strategy in {"public", "aligned"}:
        public_validation_examples = build_public_training_examples(
            args.public_set,
            args.public_top50,
            args.catalog,
            negatives_per_positive=args.negatives_per_positive,
            negative_pool_csv_path=args.synthetic_products,
            seed=args.seed,
            positive_weight=1.0,
            negative_weight=1.0,
        )
    train_examples, valid_examples = _split_examples(examples, args.eval_fraction, args.seed)
    distribution = summarize_examples(examples)
    summary = {
        "data_strategy": args.data_strategy,
        "positive_examples": sum(example.label == 1.0 for example in examples),
        "negative_examples": sum(example.label == 0.0 for example in examples),
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "public_validation_examples": len(public_validation_examples),
        "trainable_parameters": None,
        "best_public_accuracy": None,
        "selected_epoch": None,
        "distribution": distribution.__dict__,
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, trainable_count = _load_model(args.base_model, device)
    summary["trainable_parameters"] = trainable_count
    print(json.dumps({"trainable_parameters": trainable_count, "device": device}))

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    best_public_accuracy = -1.0
    best_state_dict = None
    selected_epoch: int | None = None

    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(train_examples)
        total_loss = 0.0
        total = 0
        correct = 0
        model.train()
        for start in range(0, len(train_examples), args.batch_size):
            batch = train_examples[start : start + args.batch_size]
            inputs, labels, weights = _encode_batch(tokenizer, batch, device, args.max_length)
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
            weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits.squeeze(-1)
            elementwise_loss = loss_fn(logits, labels_tensor)
            loss = (elementwise_loss * weights_tensor).mean()
            loss.backward()
            optimizer.step()
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((predictions == labels_tensor).sum().item())
            total += len(batch)
            total_loss += float(loss.item()) * len(batch)
        train_accuracy = correct / max(1, total)
        valid_accuracy = evaluate(
            tokenizer, model, valid_examples, device, args.max_length, args.batch_size
        )
        public_accuracy = evaluate(
            tokenizer, model, public_validation_examples, device, args.max_length, args.batch_size
        )
        if public_validation_examples and public_accuracy >= best_public_accuracy:
            best_public_accuracy = public_accuracy
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            selected_epoch = epoch
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(total_loss / max(1, total), 6),
                    "train_accuracy": round(train_accuracy, 6),
                    "valid_accuracy": round(valid_accuracy, 6),
                    "public_accuracy": round(public_accuracy, 6),
                }
            )
        )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    summary["best_public_accuracy"] = round(best_public_accuracy, 6)
    summary["selected_epoch"] = selected_epoch

    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.output)
    model.save_pretrained(args.output)
    (args.output / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved checkpoint to {args.output}")


def evaluate(tokenizer, model, examples, device, max_length, batch_size) -> float:
    import torch

    if not examples:
        return 0.0
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            inputs, labels, _ = _encode_batch(tokenizer, batch, device, max_length)
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
            logits = model(**inputs).logits.squeeze(-1)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((predictions == labels_tensor).sum().item())
            total += len(batch)
    return correct / max(1, total)


if __name__ == "__main__":
    train(parse_args())

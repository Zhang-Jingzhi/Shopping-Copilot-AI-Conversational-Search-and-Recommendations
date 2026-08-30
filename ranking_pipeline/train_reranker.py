"""Train a lightweight Qwen3-Reranker scoring head on the public 200 sessions.

The official rules allow local scoring logic and prompt tuning, while
full-parameter training of base foundational LLMs is out of scope. This script
therefore freezes the transformer trunk and trains only the final
classification/score parameters.
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
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_ROOT = REPOSITORY_ROOT / "retrieval-and-reranking"
DATA_ROOT = RETRIEVAL_ROOT / "data"
DEFAULT_PUBLIC_SET = DATA_ROOT / "public_set.jsonl"
DEFAULT_TOP50 = DATA_ROOT / "techjam-precomputed-rankings-200-and-3021" / "public200_top50.jsonl"
DEFAULT_CATALOG = DATA_ROOT / "catalog.jsonl"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "ranking_pipeline" / "checkpoints" / "qwen3-reranker-0.6b-shopping"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-set", type=Path, default=DEFAULT_PUBLIC_SET)
    parser.add_argument("--public-top50", type=Path, default=DEFAULT_TOP50)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--negatives-per-positive", type=int, default=4)
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
    for example in examples:
        key = (example.query, example.parent_asin if example.label == 1.0 else "")
        if example.label == 1.0:
            group_id = str(len(order))
            order.append(group_id)
            positive_groups[group_id] = [example]
            last_group = group_id
        else:
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
    return (
        {name: value.to(device) for name, value in inputs.items()},
        labels,
    )


def train(args: argparse.Namespace) -> None:
    import torch
    from torch import nn

    examples = build_public_training_examples(
        args.public_set,
        args.public_top50,
        args.catalog,
        negatives_per_positive=args.negatives_per_positive,
        seed=args.seed,
        limit=args.limit,
    )
    if not examples:
        raise RuntimeError("No public training examples could be built")
    train_examples, valid_examples = _split_examples(examples, args.eval_fraction, args.seed)
    summary = {
        "positive_examples": sum(example.label == 1.0 for example in examples),
        "negative_examples": sum(example.label == 0.0 for example in examples),
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "trainable_parameters": None,
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
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(train_examples)
        total_loss = 0.0
        total = 0
        correct = 0
        model.train()
        for start in range(0, len(train_examples), args.batch_size):
            batch = train_examples[start : start + args.batch_size]
            inputs, labels = _encode_batch(tokenizer, batch, device, args.max_length)
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits.squeeze(-1)
            loss = loss_fn(logits, labels_tensor)
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
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": round(total_loss / max(1, total), 6),
                    "train_accuracy": round(train_accuracy, 6),
                    "valid_accuracy": round(valid_accuracy, 6),
                }
            )
        )

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
            inputs, labels = _encode_batch(tokenizer, batch, device, max_length)
            labels_tensor = torch.tensor(labels, dtype=torch.float32, device=device)
            logits = model(**inputs).logits.squeeze(-1)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((predictions == labels_tensor).sum().item())
            total += len(batch)
    return correct / max(1, total)


if __name__ == "__main__":
    train(parse_args())

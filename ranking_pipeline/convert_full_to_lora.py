"""Convert an existing full Qwen3 checkpoint to a PEFT LoRA adapter.

The old training loop saved the complete frozen backbone plus the trained
``score`` head. This script keeps the trained score head and wraps the base
model with LoRA, then saves only the adapter files (``adapter_model.safetensors``
and ``adapter_config.json``). Because the old loop froze the backbone, the
converted adapter reproduces the same effective checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ranking_pipeline.qwen_reranker import DEFAULT_MODEL
from ranking_pipeline.train_reranker import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT
    / "ranking_pipeline"
    / "checkpoints"
    / "qwen3-reranker-0.6B-shopping"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "ranking_pipeline"
    / "checkpoints"
    / "qwen3-reranker-0.6B-shopping-lora"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def copy_score_head(source_model, target_model) -> None:
    source_score = getattr(source_model, "score", None)
    target_score = getattr(target_model, "score", None)
    if source_score is None or target_score is None:
        raise RuntimeError("Both models must expose a sequence-classification score head")
    target_score.load_state_dict(source_score.state_dict())


def convert(checkpoint: Path, output: Path, base_model: str, device: str | None) -> dict:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    full_model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint))
    base_model_instance = AutoModelForSequenceClassification.from_pretrained(base_model)
    copy_score_head(full_model, base_model_instance)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=list(LORA_TARGET_MODULES),
        modules_to_save=["score"],
    )
    peft_model = get_peft_model(base_model_instance, lora_config).to(device)
    output.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(output))

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(str(output))

    adapter_file = output / "adapter_model.safetensors"
    summary = {
        "source_checkpoint": str(checkpoint),
        "base_model": base_model,
        "lora": {
            "r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_modules": list(LORA_TARGET_MODULES),
            "modules_to_save": ["score"],
        },
        "adapter_bytes": adapter_file.stat().st_size if adapter_file.exists() else 0,
    }
    (output / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = convert(args.checkpoint, args.output, args.base_model, args.device)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Train a listwise MRR reranker using the locked-exact ordering as a teacher.

This is intentionally separate from ``train_reranker.py``. The existing script
trains a pointwise BCE head (or a softmax listwise head), but its inference path
still exposes raw per-document sigmoid scores. This script builds query-level
groups from the locked-exact retrieval pipeline, treats the locked-exact order
as a teacher permutation, and optimizes a rank-sensitive MRR surrogate.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ranking_pipeline.training_data import (
    RerankTrainingExample,
    intent_card,
    load_catalog,
    load_jsonl,
    product_text,
    query_text,
    requirements_from_product,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_ROOT = REPOSITORY_ROOT / "retrieval-and-reranking"
DATA_ROOT = RETRIEVAL_ROOT / "data"
RESULTS_ROOT = REPOSITORY_ROOT / "ranking_pipeline" / "results"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "ranking_pipeline" / "checkpoints"

DEFAULT_PUBLIC_SET = DATA_ROOT / "public_set.jsonl"
DEFAULT_CATALOG = DATA_ROOT / "catalog.jsonl"
DEFAULT_SYNTHETIC_SET = (
    REPOSITORY_ROOT / "synthetic-data-3021" / "data" / "synthetic_contract_matched_all_3021.jsonl"
)
DEFAULT_SYNTHETIC_TIERS = (
    REPOSITORY_ROOT / "synthetic-data-3021" / "data" / "synthetic_contract_matched_all_3021_tiers.jsonl"
)
DEFAULT_BASE_MODEL = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
DEFAULT_ADAPTER = CHECKPOINT_ROOT / "0.6Blora_aligned_from_shopping_lora_epoch1"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class MRRGroup:
    """A query-level ranking group derived from one public session."""

    sample_id: str
    scenario_type: str
    target_id: str
    candidate_ids: tuple[str, ...]
    teacher_order: tuple[str, ...]
    locked_rank: int
    locked_reciprocal_rank: float
    requirements: dict[str, Any]
    user_profile: dict[str, Any]
    source: str = "public"
    weight: float = 1.0


def _requirements_to_dict(requirements: Any) -> dict[str, Any]:
    return {
        "category": getattr(requirements, "category", ""),
        "hard_constraints": tuple(getattr(requirements, "hard_constraints", ())),
        "soft_preferences": tuple(getattr(requirements, "soft_preferences", ())),
    }


def _requirements_from_dict(payload: Mapping[str, Any]) -> Any:
    from techjam_agent.contracts import Requirements

    return Requirements(
        category=str(payload.get("category") or ""),
        hard_constraints=tuple(payload.get("hard_constraints") or ()),
        soft_preferences=tuple(payload.get("soft_preferences") or ()),
    )


def _candidate_from_product(parent_asin: str, product: Mapping[str, Any], *, rank: int) -> Any:
    from techjam_agent.contracts import Candidate

    public_product = {
        field: product.get(field)
        for field in ("title", "categories", "features", "details", "description", "store")
    }
    return Candidate(
        parent_asin=parent_asin,
        candidate_rank=rank,
        source_ranks={"train": rank},
        product=public_product,
    )


def _make_requirements_for_sample(sample: Mapping[str, Any], target_product: Mapping[str, Any]) -> Any:
    from techjam_agent.contracts import Requirements

    base = requirements_from_product(target_product)
    if str(sample.get("scenario_type")) != "intent_override":
        return base

    card = intent_card(target_product)
    hard = card.get("hard_constraints") or []
    soft = card.get("soft_preferences") or []
    old_value = str(soft[-1] if soft else "I prefer a different style.")
    new_value = str(hard[0] if hard else "Please prioritize the target requirements.")
    return Requirements(
        category=base.category,
        hard_constraints=tuple(dict.fromkeys((*base.hard_constraints, new_value))),
        soft_preferences=tuple(
            value for value in base.soft_preferences if value not in {old_value, new_value}
        ),
    )


def _locked_exact_teacher(
    requirements: Any,
    target_id: str,
    *,
    catalog: Mapping[str, Mapping[str, Any]],
    group_size: int,
    generator: Any | None = None,
) -> tuple[list[str], int, float]:
    from techjam_agent.ranking import LockedWeightedRrfTop10Reranker
    from techjam_agent.retrieval import ExactDenseTop50CandidateGenerator

    if generator is None:
        generator = ExactDenseTop50CandidateGenerator(str(DATA_ROOT / "catalog.jsonl"))
    candidate_set = generator.generate(requirements, session_id="mrr-teacher", turn=3)
    candidates = list(candidate_set.candidates)
    candidate_ids = [candidate.parent_asin for candidate in candidates]
    if target_id not in candidate_ids:
        target_product = catalog.get(target_id)
        if target_product is None:
            raise ValueError(f"Target {target_id} is missing from the catalog")
        candidates.append(
            _candidate_from_product(target_id, target_product, rank=len(candidates) + 1)
        )
        candidate_ids.append(target_id)

    candidate_set = type(candidate_set)(
        candidate_set_id=candidate_set.candidate_set_id,
        session_id=candidate_set.session_id,
        turn=candidate_set.turn,
        requirements=requirements,
        candidates=tuple(candidates),
    )
    locked_result = LockedWeightedRrfTop10Reranker().rerank(
        candidate_set,
        top_k=len(candidates),
    )
    locked_order = [ranked.parent_asin for ranked in locked_result.ranked_candidates]
    if target_id not in locked_order:
        raise RuntimeError(f"Target {target_id} was not produced by locked-exact")
    locked_rank = locked_order.index(target_id) + 1
    locked_rr = 1.0 / locked_rank
    teacher_order = [target_id] + [
        parent_asin for parent_asin in locked_order if parent_asin != target_id
    ]
    if group_size and group_size < len(teacher_order):
        teacher_order = teacher_order[:group_size]
    return teacher_order, locked_rank, locked_rr


def _build_groups_from_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    catalog: Mapping[str, Mapping[str, Any]],
    generator: Any,
    group_size: int,
    limit: int | None,
    source: str,
    weight: float,
) -> tuple[list[MRRGroup], dict[str, Any]]:
    groups: list[MRRGroup] = []
    diagnostics: dict[str, Any] = {
        "source": source,
        "sample_count": 0,
        "locked_rank_distribution": {},
    }
    for sample in samples:
        sample_id = str(sample["sample_id"])
        target_id = str(sample["ground_truth"]["parent_asin"])
        target_product = catalog.get(target_id)
        if target_product is None:
            continue
        requirements = _make_requirements_for_sample(sample, target_product)
        teacher_order, locked_rank, locked_rr = _locked_exact_teacher(
            requirements,
            target_id,
            catalog=catalog,
            group_size=group_size,
            generator=generator,
        )
        groups.append(
            MRRGroup(
                sample_id=sample_id,
                scenario_type=str(sample.get("scenario_type") or ""),
                target_id=target_id,
                candidate_ids=tuple(teacher_order),
                teacher_order=tuple(teacher_order),
                locked_rank=locked_rank,
                locked_reciprocal_rank=locked_rr,
                requirements=_requirements_to_dict(requirements),
                user_profile=dict(sample.get("user_profile") or {}),
                source=source,
                weight=weight,
            )
        )
        diagnostics["sample_count"] += 1
        rank_bucket = "1" if locked_rank == 1 else ("2-5" if locked_rank <= 5 else "6-10" if locked_rank <= 10 else "11+")
        diagnostics["locked_rank_distribution"][rank_bucket] = (
            diagnostics["locked_rank_distribution"].get(rank_bucket, 0) + 1
        )
        if limit is not None and len(groups) >= limit:
            break
    return groups, diagnostics


def build_public_mrr_groups(
    *,
    public_set_path: Path,
    catalog_path: Path,
    group_size: int = 20,
    limit: int | None = None,
    weight: float = 5.0,
) -> tuple[list[MRRGroup], dict[str, Any]]:
    samples = load_jsonl(public_set_path)
    catalog = load_catalog(catalog_path)
    from techjam_agent.retrieval import ExactDenseTop50CandidateGenerator

    generator = ExactDenseTop50CandidateGenerator(str(catalog_path))
    return _build_groups_from_samples(
        samples,
        catalog=catalog,
        generator=generator,
        group_size=group_size,
        limit=limit,
        source="public",
        weight=weight,
    )


def build_synthetic_mrr_groups(
    *,
    synthetic_set_path: Path,
    synthetic_tiers_path: Path,
    catalog_path: Path,
    group_size: int = 20,
    limit: int | None = 300,
    weight: float = 1.0,
    retrieval_mode: str = "lite",
    tier_filter: Sequence[str] = ("high_confidence", "probable"),
) -> tuple[list[MRRGroup], dict[str, Any]]:
    samples = load_jsonl(synthetic_set_path)
    tiers = {
        str(row["sample_id"]): str(row.get("quality_tier") or "")
        for row in load_jsonl(synthetic_tiers_path)
    }
    allowed_tiers = set(tier_filter)
    samples = [
        sample
        for sample in samples
        if str(sample.get("sample_id")) in tiers
        and tiers[str(sample.get("sample_id"))] in allowed_tiers
    ]
    catalog = load_catalog(catalog_path)
    if retrieval_mode == "exact":
        from techjam_agent.retrieval import ExactDenseTop50CandidateGenerator

        generator = ExactDenseTop50CandidateGenerator(str(catalog_path))
    else:
        from techjam_agent.retrieval import LiteTop50CandidateGenerator

        generator = LiteTop50CandidateGenerator(str(catalog_path))
    return _build_groups_from_samples(
        samples,
        catalog=catalog,
        generator=generator,
        group_size=group_size,
        limit=limit,
        source="synthetic",
        weight=weight,
    )


def _group_to_examples(
    group: MRRGroup,
    *,
    catalog: Mapping[str, Mapping[str, Any]],
) -> list[RerankTrainingExample]:
    requirements = _requirements_from_dict(group.requirements)
    query = query_text(requirements, user_profile=group.user_profile)
    examples: list[RerankTrainingExample] = []
    for rank, parent_asin in enumerate(group.candidate_ids, start=1):
        product = catalog.get(parent_asin)
        if product is None:
            continue
        examples.append(
            RerankTrainingExample(
                query=query,
                document=product_text(_candidate_from_product(parent_asin, product, rank=rank)),
                label=1.0 if parent_asin == group.target_id else 0.0,
                parent_asin=parent_asin,
                group_id=group.sample_id,
                weight=group.weight,
                source=group.source,
                tier=group.source,
            )
        )
    return examples


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-strategy", choices=("public", "aligned"), default="aligned")
    parser.add_argument("--public-set", type=Path, default=DEFAULT_PUBLIC_SET)
    parser.add_argument("--synthetic-set", type=Path, default=DEFAULT_SYNTHETIC_SET)
    parser.add_argument("--synthetic-tiers", type=Path, default=DEFAULT_SYNTHETIC_TIERS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-checkpoint", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--loss", choices=("listnet", "listmle"), default="listmle")
    parser.add_argument("--locked-reward-scale", type=float, default=0.25)
    parser.add_argument("--public-weight", type=float, default=5.0)
    parser.add_argument("--synthetic-weight", type=float, default=1.0)
    parser.add_argument("--synthetic-limit", type=int, default=300)
    parser.add_argument("--synthetic-retrieval-mode", choices=("lite", "exact"), default="lite")
    parser.add_argument("--synthetic-tier-filter", nargs="+", default=("high_confidence", "probable"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache-groups", type=Path, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--no-load-in-4bit", action="store_false", dest="load_in_4bit")
    parser.set_defaults(load_in_4bit=True)
    return parser.parse_args()


def _build_output(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return CHECKPOINT_ROOT / (
        f"0.6Blora_mrr_locked_{args.data_strategy}_{args.loss}_g{args.group_size}_{timestamp}"
    )


def _encode_group(
    tokenizer: Any,
    examples: Sequence[RerankTrainingExample],
    device: str,
    max_length: int,
) -> dict[str, Any]:
    from ranking_pipeline.qwen_reranker import format_pair

    texts = [format_pair(example.query, example.document) for example in examples]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation="longest_first",
        max_length=max_length,
    )
    return {name: value.to(device) for name, value in inputs.items()}


def _listnet_loss(logits: Any, target_index: int) -> Any:
    import torch

    return -torch.log_softmax(logits, dim=0)[target_index]


def _listmle_loss(logits: Any, teacher_order: Sequence[str], id_to_index: Mapping[str, int]) -> Any:
    import torch

    remaining = list(range(logits.shape[0]))
    total = torch.zeros((), device=logits.device)
    for parent_asin in teacher_order[:-1]:
        if parent_asin not in id_to_index:
            continue
        index = id_to_index[parent_asin]
        if index not in remaining:
            continue
        log_probs = torch.log_softmax(logits[remaining], dim=0)
        relative_index = remaining.index(index)
        total = total - log_probs[relative_index]
        remaining.remove(index)
    return total


def _group_mrr(logits: Any, target_index: int) -> float:
    import torch

    order = torch.argsort(logits, descending=True).tolist()
    rank = order.index(target_index) + 1
    return 1.0 / rank


def _save_checkpoint(model: Any, tokenizer: Any, output: Path, summary: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output)
    model.save_pretrained(output)
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved checkpoint to {output}")


def _load_peft_model(
    base_model: str,
    device: str,
    *,
    load_in_4bit: bool,
    adapter_checkpoint: Path | None,
) -> tuple[Any, Any, int]:
    import torch
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {}
    if load_in_4bit:
        if not device.startswith("cuda"):
            raise RuntimeError("--load-in-4bit requires a CUDA device. Use --no-load-in-4bit for CPU.")
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["device_map"] = "auto"
    model = AutoModelForSequenceClassification.from_pretrained(base_model, **model_kwargs)
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=list(LORA_TARGET_MODULES),
        modules_to_save=["score"],
    )
    if adapter_checkpoint is not None:
        adapter_checkpoint = Path(adapter_checkpoint)
        if not adapter_checkpoint.is_dir():
            raise ValueError(f"--adapter-checkpoint must be a directory: {adapter_checkpoint}")
        if not (adapter_checkpoint / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"Adapter checkpoint is missing adapter_config.json: {adapter_checkpoint}"
            )
        if not (adapter_checkpoint / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(
                f"Adapter checkpoint is missing adapter_model.safetensors: {adapter_checkpoint}"
            )
        model = PeftModel.from_pretrained(model, adapter_checkpoint, is_trainable=True)
    else:
        model = get_peft_model(model, lora_config)
    model.to(device)
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    model.train()
    return tokenizer, model, trainable_count


def _evaluate_groups(
    tokenizer: Any,
    model: Any,
    groups: Sequence[MRRGroup],
    catalog: Mapping[str, Mapping[str, Any]],
    device: str,
    max_length: int,
) -> dict[str, float]:
    import torch

    if not groups:
        return {"mrr": 0.0, "top1_accuracy": 0.0}
    model.eval()
    reciprocal_ranks: list[float] = []
    correct_top1 = 0
    with torch.inference_mode():
        total_groups = len(groups)
        eval_started_at = time.perf_counter()
        for eval_index, group in enumerate(groups, start=1):
            examples = _group_to_examples(group, catalog=catalog)
            if not examples:
                continue
            inputs = _encode_group(tokenizer, examples, device, max_length)
            logits = model(**inputs).logits.squeeze(-1)
            target_index = next(
                (
                    index
                    for index, example in enumerate(examples)
                    if example.parent_asin == group.target_id
                ),
                None,
            )
            if target_index is None:
                continue
            order = torch.argsort(logits, descending=True).tolist()
            rank = order.index(target_index) + 1
            reciprocal_ranks.append(1.0 / rank)
            correct_top1 += int(order[0] == target_index)
            if eval_index % 50 == 0 or eval_index == total_groups:
                print(
                    json.dumps(
                        {
                            "phase": "evaluate_groups",
                            "step": eval_index,
                            "steps": total_groups,
                            "running_mrr": round(sum(reciprocal_ranks) / max(1, len(reciprocal_ranks)), 6),
                            "running_top1_accuracy": round(correct_top1 / max(1, len(reciprocal_ranks)), 6),
                            "elapsed_seconds": round(time.perf_counter() - eval_started_at, 1),
                        }
                    ),
                    flush=True,
                )
    return {
        "mrr": sum(reciprocal_ranks) / max(1, len(reciprocal_ranks)),
        "top1_accuracy": correct_top1 / max(1, len(reciprocal_ranks)),
    }


def _train(args: argparse.Namespace, groups: list[MRRGroup], catalog: Mapping[str, Mapping[str, Any]]) -> None:
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, trainable_count = _load_peft_model(
        args.base_model,
        device,
        load_in_4bit=args.load_in_4bit,
        adapter_checkpoint=args.adapter_checkpoint,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    run_output = _build_output(args)
    best_mrr = -1.0
    best_state_dict = None
    selected_epoch: int | None = None
    summary: dict[str, Any] = {
        "task": "mrr-listwise",
        "data_strategy": args.data_strategy,
        "loss": args.loss,
        "group_size": args.group_size,
        "group_count": len(groups),
        "public_group_count": sum(group.source == "public" for group in groups),
        "synthetic_group_count": sum(group.source == "synthetic" for group in groups),
        "public_weight": args.public_weight,
        "synthetic_weight": args.synthetic_weight,
        "trainable_parameters": trainable_count,
        "base_model": args.base_model,
        "adapter_checkpoint": str(args.adapter_checkpoint),
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "locked_reward_scale": args.locked_reward_scale,
        "best_group_mrr": None,
        "selected_epoch": None,
    }

    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(groups)
        model.train()
        epoch_started_at = time.perf_counter()
        total_loss = 0.0
        for step, group in enumerate(groups, start=1):
            examples = _group_to_examples(group, catalog=catalog)
            if not examples:
                continue
            target_index = next(
                index
                for index, example in enumerate(examples)
                if example.parent_asin == group.target_id
            )
            inputs = _encode_group(tokenizer, examples, device, args.max_length)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits.squeeze(-1)
            if args.loss == "listmle":
                id_to_index = {
                    example.parent_asin: index
                    for index, example in enumerate(examples)
                }
                loss = _listmle_loss(logits, group.teacher_order, id_to_index)
            else:
                loss = _listnet_loss(logits, target_index)
            reward_weight = 1.0 + args.locked_reward_scale * (
                1.0 - group.locked_reciprocal_rank
            )
            loss = loss * reward_weight * group.weight
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            if args.log_interval > 0 and step % args.log_interval == 0:
                now = time.perf_counter()
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "steps": len(groups),
                            "step_loss": round(float(loss.item()), 6),
                            "group_mrr": round(_group_mrr(logits, target_index), 6),
                            "locked_rank": group.locked_rank,
                            "reward_weight": round(reward_weight, 4),
                            "elapsed_seconds": round(now - epoch_started_at, 1),
                        }
                    ),
                    flush=True,
                )
        metrics = _evaluate_groups(
            tokenizer,
            model,
            groups,
            catalog,
            device,
            args.max_length,
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "mean_loss": round(total_loss / max(1, len(groups)), 6),
                    "model_group_mrr": round(metrics["mrr"], 6),
                    "model_top1_accuracy": round(metrics["top1_accuracy"], 6),
                    "epoch_seconds": round(time.perf_counter() - epoch_started_at, 1),
                }
            ),
            flush=True,
        )
        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in get_peft_model_state_dict(model).items()
            }
            selected_epoch = epoch

    if best_state_dict is not None:
        set_peft_model_state_dict(model, best_state_dict)
    summary["best_group_mrr"] = round(best_mrr, 6)
    summary["selected_epoch"] = selected_epoch
    _save_checkpoint(model, tokenizer, run_output, summary)
    print(json.dumps({"output": str(run_output), "best_group_mrr": round(best_mrr, 6), "selected_epoch": selected_epoch}))


def _load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_evaluation(
    model_path: Path,
    *,
    output_path: Path,
    mode: str = "local",
    retrieval_mode: str = "exact",
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "ranking_pipeline.evaluate_agent",
        "--mode",
        mode,
        "--retrieval-mode",
        retrieval_mode,
        "--reranker-model",
        str(model_path),
        "--output",
        str(output_path),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return _load_result(output_path)


def _summarise_metric(path: Path, label: str) -> dict[str, Any]:
    data = _load_result(path)
    override = data.get("scenario_metrics", {}).get("intent_override", {})
    return {
        "label": label,
        "file": path.name,
        "hit_rate_at_10": data.get("hit_rate_at_10"),
        "mrr": data.get("mrr"),
        "mttc": data.get("mttc"),
        "technical_score": data.get("recommended_technical_score"),
        "intent_override_mrr": override.get("mrr"),
        "intent_override_hit": override.get("hit_rate_at_10"),
    }


def _compare_with_baseline(
    new_model_path: Path,
    baseline_model_path: Path,
    *,
    results_root: Path = RESULTS_ROOT,
) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    baseline_output = results_root / "local-exact-mrr-baseline.json"
    new_output = results_root / "local-exact-mrr-locked-teacher.json"
    locked_output = results_root / "locked-exact-mrr-baseline.json"

    _run_evaluation(baseline_model_path, output_path=baseline_output)
    _run_evaluation(new_model_path, output_path=new_output)
    _run_evaluation(
        baseline_model_path,
        output_path=locked_output,
        mode="locked",
    )

    comparison = {
        "locked_exact": _summarise_metric(locked_output, "locked-exact"),
        "baseline_local_exact": _summarise_metric(
            baseline_output,
            "baseline-local-exact",
        ),
        "trained_local_exact": _summarise_metric(
            new_output,
            "trained-local-exact",
        ),
    }
    comparison_path = results_root / "mrr-locked-teacher-comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, indent=2))


def main() -> None:
    args = _parse_args()
    if args.group_size <= 1:
        raise ValueError("--group-size must be at least 2")
    if args.adapter_checkpoint is not None and "mrr_locked_exact" in args.adapter_checkpoint.name:
        raise SystemExit(
            "Refusing to continue from a public-200-only MRR checkpoint. "
            "Use --adapter-checkpoint with a clean adapter such as "
            "ranking_pipeline/checkpoints/0.6Blora_aligned_from_shopping_lora_epoch1."
        )
    if args.cache_groups is not None and args.cache_groups.is_file():
        cached = json.loads(args.cache_groups.read_text(encoding="utf-8"))
        groups = [
            MRRGroup(
                sample_id=item["sample_id"],
                scenario_type=item["scenario_type"],
                target_id=item["target_id"],
                candidate_ids=tuple(item["candidate_ids"]),
                teacher_order=tuple(item["teacher_order"]),
                locked_rank=int(item["locked_rank"]),
                locked_reciprocal_rank=float(item["locked_reciprocal_rank"]),
                requirements=item["requirements"],
                user_profile=item["user_profile"],
                source=str(item.get("source") or "public"),
                weight=float(item.get("weight") or 1.0),
            )
            for item in cached["groups"]
        ]
        diagnostics = cached["diagnostics"]
    else:
        public_groups, public_diagnostics = build_public_mrr_groups(
            public_set_path=args.public_set,
            catalog_path=args.catalog,
            group_size=args.group_size,
            limit=args.limit,
            weight=args.public_weight,
        )
        groups = list(public_groups)
        diagnostics = dict(public_diagnostics)
        if args.data_strategy == "aligned":
            synthetic_groups, synthetic_diagnostics = build_synthetic_mrr_groups(
                synthetic_set_path=args.synthetic_set,
                synthetic_tiers_path=args.synthetic_tiers,
                catalog_path=args.catalog,
                group_size=args.group_size,
                limit=args.synthetic_limit,
                weight=args.synthetic_weight,
                retrieval_mode=args.synthetic_retrieval_mode,
                tier_filter=tuple(args.synthetic_tier_filter),
            )
            groups.extend(synthetic_groups)
            diagnostics = {
                "public": public_diagnostics,
                "synthetic": synthetic_diagnostics,
                "total_groups": len(groups),
            }
        if args.cache_groups is not None:
            args.cache_groups.parent.mkdir(parents=True, exist_ok=True)
            args.cache_groups.write_text(
                json.dumps(
                    {
                        "diagnostics": diagnostics,
                        "groups": [
                            {
                                "sample_id": group.sample_id,
                                "scenario_type": group.scenario_type,
                                "target_id": group.target_id,
                                "candidate_ids": list(group.candidate_ids),
                                "teacher_order": list(group.teacher_order),
                                "locked_rank": group.locked_rank,
                                "locked_reciprocal_rank": group.locked_reciprocal_rank,
                                "requirements": group.requirements,
                                "user_profile": group.user_profile,
                                "source": group.source,
                                "weight": group.weight,
                            }
                            for group in groups
                        ],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    print(json.dumps(diagnostics, indent=2))
    if args.dry_run:
        for group in groups[:3]:
            print(
                json.dumps(
                    {
                        "sample_id": group.sample_id,
                        "scenario_type": group.scenario_type,
                        "target_id": group.target_id,
                        "locked_rank": group.locked_rank,
                        "group_size": len(group.candidate_ids),
                    },
                    indent=2,
                )
            )
        return

    catalog = load_catalog(args.catalog)
    _train(args, groups, catalog)
    run_output = _build_output(args)
    if args.evaluate:
        if args.adapter_checkpoint is None:
            raise SystemExit("--evaluate requires --adapter-checkpoint for the clean baseline model")
        _compare_with_baseline(run_output, args.adapter_checkpoint)


if __name__ == "__main__":
    main()

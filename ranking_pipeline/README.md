# Ranking Pipeline

This directory implements component 4 and D:

- semantic reranking of the `CandidateSet` returned by retrieval
- hard-constraint filtering and a locked weighted-RRF pre-rank
- `Qwen3-Reranker-0.6B` local pointwise scoring on the Top20 candidate subset
- locked weighted-RRF + pointwise-score fusion, with deterministic fallback
- structured per-candidate scores/reasons and dynamic locked/hybrid routing
- over-general candidate-pool detection and attribute-aware clarification policy
- reproducible public-set training and full-pipeline evaluation

## Layout

```text
ranking_pipeline/
  context.py            profile compaction and intent-override parsing
  memory_context.py     intent-router/state-memory adapters
  prompt.py             compact listwise prompt and strict JSON parser
  qwen_reranker.py      Qwen3-Reranker-0.6B pointwise adapter
  contextual_ranking.py hard filter + pre-rank + final reranker + policy
  training_data.py      public/synthetic pair construction
  distribution_alignment.py aligned public+3021 training data
  train_reranker.py     scoring-head training script
  evaluate_agent.py     official evaluator wrapper
  analyze_cutoff.py     offline Top20/Top50 recall check
  diagnose_synthetic.py synthetic 3,021 diagnostic summarizer
  summarize_results.py  saved-result comparison table
  checkpoints/          trained scoring head
  results/              local evaluator results
```

`ranking_pipeline` is a sibling of `retrieval-and-reranking`. Import path
bootstrapping is kept in `retrieval-and-reranking/techjam_agent/__init__.py`
and this package's `__init__.py`, so both the official evaluator and this
package can be launched from their natural working directories without setting
`PYTHONPATH`.

## Model Decision

Use `Qwen3-Reranker-0.6B`, not `Qwen3-VL-Reranker-2B`.

The catalog and simulated dialogue are text only. The VL reranker adds a vision
backbone that is unused here, so it only increases memory and latency. The
0.6B text model is small enough to run offline on a 6GB consumer GPU and is
also usable as a CPU fallback with reduced throughput.

The default checkpoint is `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`, a
sequence-classification conversion of `Qwen/Qwen3-Reranker-0.6B`. This lets the
same `AutoModelForSequenceClassification` API be used for zero-shot inference
and scoring-head training.

## Top20 vs Top50

Feed the locked pre-ranker's Top20, not the original Top50, and not the first
20 rows of the raw Top50 list.

On the public precomputed files:

```json
{
  "sample_count": 200,
  "top50_coverage": "199/200",
  "raw_top50_first20_coverage": "192/200",
  "top10_hit": "198/200",
  "top20_coverage_lower_bound_from_locked_top10": "198/200",
  "mrr_at_10": 0.921843
}
```

The raw first-20 slice loses targets compared with the locked Top10. The locked
Top10 covers 198/200, so the same locked ordering truncated to Top20 is the
correct final-model input. Using all 50 only increases prompt length, latency,
and small-model ranking noise.

## Install

From the repository root:

```powershell
python -m pip install torch transformers
python -m unittest discover -s ranking_pipeline/tests -t . -v
```

The competition catalog is not committed. Download and verify it from the
official `participant-kit` GitHub Release into
`retrieval-and-reranking/data/catalog.jsonl`.

## Train The Scoring Head

The official scope permits local scoring logic and prompt tuning, while
full-parameter training of base foundation models is out of scope. This script
freezes the transformer trunk and trains only the `score` classifier parameter.

The default `aligned` strategy combines the public 200 gold set with the
synthetic 3,021 proxy set. The synthetic proxy is public-schema compatible but
not the organizer private-800 set: use it only for distribution alignment and
calibration, never as the final reported score. Synthetic positives use target
products as positives, while negatives are sampled from the same leaf category
with weights derived from `quality_tier` and `selection_frequency`.

For the joint strategy, public positives default to `5.0` loss weight (the
recommended 5-10x band), and each epoch is evaluated on the public set. The
best public-accuracy checkpoint is restored before saving. A public-only run
defaults to one epoch; an aligned run defaults to two epochs.

```powershell
python -m ranking_pipeline.train_reranker `
  --data-strategy aligned `
  --epochs 2 `
  --batch-size 4 `
  --negatives-per-positive 4 `
  --synthetic-negatives-per-positive 4 `
  --max-length 512 `
  --output ranking_pipeline\checkpoints\qwen3-reranker-0.6B-shopping
```

Set `--data-strategy public` for the previous public-200-only run, or
`--data-strategy synthetic` for a synthetic-only dry run/ablation. The old
`techjam-precomputed-rankings-200-and-3021` directory is no longer required:
pass `--public-top50` only when a precomputed public Top50 file is available;
otherwise the builder samples negatives from the synthetic product CSV/catalog.

First run `--dry-run` to inspect the generated pairs without downloading model
weights.

After generating synthetic ranked predictions, emit diagnostics only:

```powershell
python -m ranking_pipeline.diagnose_synthetic `
  --rankings ranking_pipeline\results\synthetic-3021-ranked.jsonl `
  --top-k 10
```

The diagnostic report contains per-tier recall, score distribution, and
over-general rate. Do not use it as the official evaluation metric.

The precomputed Top50 diagnostic output is saved to
`ranking_pipeline/results/synthetic-3021-diagnostics.json` and records
`per-tier recall` from `0.950` to `0.986`, a `43.36%` over-general rate, and
the fallback score distribution across 30,210 scored candidates.

## Run The Full Pipeline

The wrapper invokes the unchanged official evaluator from
`retrieval-and-reranking`.

```powershell
# locked baseline
python -m ranking_pipeline.evaluate_agent --mode locked --retrieval-mode lite

# hard filter + locked pre-rank, no local model
python -m ranking_pipeline.evaluate_agent --mode hybrid --retrieval-mode lite

# full pipeline with the trained Qwen3-Reranker-0.6B scoring head
python -m ranking_pipeline.evaluate_agent --mode local --retrieval-mode lite

# optional: consume conversation-state-memory + intent-recognition contracts
python -m ranking_pipeline.evaluate_agent --mode local --retrieval-mode lite `
  --use-state-memory --use-intent-router
```

## Context Source

The official ``reset(session_id, user_profile)`` profile is the primary
long-term user signal. ``conversation-state-memory`` supplies the short-term
``ContextSnapshot``, and ``intent-recognition`` supplies the current-turn
``IntentResult``. ``ranking_pipeline.memory_context`` merges official profile
tags with snapshot ``profile_hints``; current-turn hard constraints still
override any profile hint.

The official requirements collector remains the retrieval/ranking requirement
source because it is the contract the evaluator already validates against. The
optional ``IntentResult`` and ``ContextSnapshot`` are supplemental inputs:
they provide current-turn route/ambiguity signals, accumulated slots,
exclusions, session summary, and profile hints. They are passed into
``set_session_context`` rather than replacing the collector output. The
long-term official profile is still the safest personalization signal; do not
let the state-memory in-memory ``UserProfile`` replace it.

The final LLM/listwise prompt is capped to ``min(top_n, max(top_k * 2, 10))``
candidates after the locked pre-rank. Prompt chars and an approximate token
count are stored on the reranker as ``last_prompt_chars`` and
``last_prompt_tokens`` for latency/telemetry verification.

Formal `exact` dense public-200 results:

| mode | Hit@10 | MRR | MTTC | Technical |
|---|---:|---:|---:|---:|
| `local-exact.json` | 0.975 | 0.786365 | 3.32 | 0.877009 |
| `locked-exact.json` | 0.970 | 0.870548 | 3.325 | 0.899664 |
| `hybrid-exact.json` | 0.835 | 0.480437 | 4.39 | 0.693831 |
| `local-exact-policy.json` | 0.850 | 0.503417 | 9.955 | 0.596925 |

The locked exact path remains the strongest overall by MRR and MTTC; local
pointwise fusion preserves Hit@10 but currently lowers MRR. Enabling the
clarification policy is not recommended for this public evaluator because it
increases MTTC sharply. A pointwise-weight sweep (`0.1`, `0.35`, `0.6`) did not
recover the locked MRR, so the locked exact path is the current recommended
submission configuration.

When CUDA reports `Unrecognized CachingAllocator option`, run exact/local
experiments with:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False'
$env:TECHJAM_RERANKER_DEVICE='cuda'
```

The policy layer selects the next `ask_attribute` instead of always returning
`other`. It accounts for prior asked attributes, missing requirement attributes,
model confidence, and the candidate pool's over-general signal. The agent keeps
this short-term clarification state and forwards it to the reranker through the
same `set_session_context` boundary used for the long-term profile.

## Unit Tests

From the repository root:

```powershell
python -m unittest discover -s ranking_pipeline/tests -t . -v
```

The ranking-pipeline suite currently contains 29 tests and does not download
model weights. The original retrieval-and-reranking suite can still be run
unchanged from that directory with:

```powershell
cd retrieval-and-reranking
python -m unittest discover -s tests -v
```

`ranking_pipeline` does not modify tracked files under
`retrieval-and-reranking`; the only integration adapter is
`ranking_pipeline/agent.py`.

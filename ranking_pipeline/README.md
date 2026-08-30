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
  prompt.py             compact listwise prompt and strict JSON parser
  qwen_reranker.py      Qwen3-Reranker-0.6B pointwise adapter
  contextual_ranking.py hard filter + pre-rank + final reranker + policy
  training_data.py      public 200 pair construction
  train_reranker.py     scoring-head training script
  evaluate_agent.py     official evaluator wrapper
  analyze_cutoff.py     offline Top20/Top50 recall check
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

```powershell
python -m ranking_pipeline.train_reranker `
  --epochs 2 `
  --batch-size 4 `
  --negatives-per-positive 2 `
  --max-length 512 `
  --output ranking_pipeline\checkpoints\qwen3-reranker-0.6B-shopping
```

First run `--dry-run` to inspect the generated pairs without downloading model
weights.

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
```

The trained-head local fusion run recorded:

```text
HitRate@10 0.975
MRR        0.878429
MTTC       3.320
Technical  0.904629
```

The locked baseline on the same `lite` retrieval path recorded
`HitRate@10 0.970`, `MRR 0.870548`, and `Technical 0.899664`; the hybrid
no-local-model run recorded `HitRate@10 0.735` and `MRR 0.302028`. The pointwise
fusion therefore preserves the locked baseline's stable ordering while using
the local model to recover some missed high-rank targets. These are full
public-evaluator smoke results, not the exact dense retrieval submission score;
`exact` still needs the BGE local assets.

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

The ranking-pipeline suite currently contains 15 tests and does not download
model weights. The original retrieval-and-reranking suite can still be run
unchanged from that directory with:

```powershell
cd retrieval-and-reranking
python -m unittest discover -s tests -v
```

`ranking_pipeline` does not modify tracked files under
`retrieval-and-reranking`; the only integration adapter is
`ranking_pipeline/agent.py`.

# Ranking Pipeline Results

Saved local-evaluator results under `ranking_pipeline/results/`.
Public-200 official local-evaluator diagnostics. The `exact` rows use
the local BGE dense embeddings and catalog assets.

## Mode Definitions

- `locked`: retrieval candidate generation plus the repository's
  `LockedWeightedRrfTop10Reranker`; no local model and no ranking-pipeline
  LLM path.
- `hybrid`: ranking-pipeline hard-constraint filter + locked weighted-RRF
  pre-rank, but no local model scoring.
- `local`: ranking-pipeline `HybridContextualReranker` with the Qwen3
  reranker pointwise scores fused with the locked pre-rank.
- `policy`: `local` plus `RecommendationClarificationPolicy` deciding
  between recommend and clarify.

`retrieval-mode=exact` means `ExactDenseTop50CandidateGenerator`, the
official dense path with BGE small embeddings. `retrieval-mode=lite` is
the local non-dense smoke path.

## Metrics

- Hit@10: target appears in the first 10 recommendations.
- MRR: mean `1 / best_rank`; zero when the target is never returned.
- MTTC: mean first-hit turn; a miss counts as `11`, so lower is better.
- Efficiency: `clamp((11 - MTTC) / 10, 0, 1)`.
- Technical: `0.50 * Hit@10 + 0.30 * MRR + 0.20 * Efficiency`.

## Key Findings

- `locked-exact` is currently the strongest submission configuration:
  Hit@10 `0.970`, MRR `0.870548`, MTTC `3.325`, Technical `0.899664`.
- `local-exact` preserves or improves Hit@10 (`0.975`) but lowers MRR
  (`0.786365`) compared with locked.
- Enabling the clarification policy sharply increases MTTC to `9.955`;
  it is not recommended for this public evaluator.
- A pointwise-weight sweep `0.1/0.35/0.6` did not recover the locked MRR.

## Synthetic 3021 Diagnostics

`synthetic-3021-diagnostics.json` reports per-tier recall, fallback score
distribution, and over-general rate for the synthetic proxy set. It is
diagnostic only and is not an official ranking score.
- Records: `3021`, Top-K: `10`, over-general rate: `0.433631`.
- Per-tier recall:
  - `high_confidence`: 1018/1065 (`0.955869`).
  - `low_likelihood`: 1562/1644 (`0.950122`).
  - `probable`: 142/144 (`0.986111`).
  - `uncertain`: 162/168 (`0.964286`).
- Fallback-score distribution: 30210 scores, mean `0.292897`, median `0.183333`.

## Result Table

| file | n | Hit@10 | MRR | MTTC | Efficiency | Technical | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| local-fusion.json | 200 | 0.975000 | 0.878429 | 3.320000 | 0.768000 | 0.904629 | override_hit=0.866667 |
| locked-exact-verify.json | 200 | 0.970000 | 0.870548 | 3.325000 | 0.767500 | 0.899664 | override_hit=0.866667 |
| locked-exact.json | 200 | 0.970000 | 0.870548 | 3.325000 | 0.767500 | 0.899664 | override_hit=0.866667 |
| locked-final.json | 200 | 0.970000 | 0.870548 | 3.325000 | 0.767500 | 0.899664 | override_hit=0.866667 |
| locked.json | 200 | 0.970000 | 0.870548 | 3.325000 | 0.767500 | 0.899664 | override_hit=0.866667 |
| local-final.json | 200 | 0.975000 | 0.790190 | 3.320000 | 0.768000 | 0.878157 | override_hit=0.866667 |
| local-exact.json | 200 | 0.975000 | 0.786365 | 3.320000 | 0.768000 | 0.877009 | override_hit=0.866667 |
| local-exact-pw060.json | 200 | 0.975000 | 0.783573 | 3.320000 | 0.768000 | 0.876172 | override_hit=0.866667 |
| local-exact-pw010.json | 200 | 0.975000 | 0.780073 | 3.320000 | 0.768000 | 0.875122 | override_hit=0.866667 |
| local-v2.json | 200 | 0.960000 | 0.701982 | 3.815000 | 0.718500 | 0.834295 | override_hit=0.866667 |
| local.json | 200 | 0.900000 | 0.684331 | 3.875000 | 0.712500 | 0.797799 | override_hit=0.766667 |
| local-v3.json | 200 | 0.830000 | 0.544861 | 4.430000 | 0.657000 | 0.709858 | override_hit=0.7 |
| hybrid-final.json | 200 | 0.835000 | 0.484262 | 4.390000 | 0.661000 | 0.694979 | override_hit=0.733333 |
| hybrid-exact.json | 200 | 0.835000 | 0.480437 | 4.390000 | 0.661000 | 0.693831 | override_hit=0.733333 |
| local-exact-policy.json | 200 | 0.850000 | 0.503417 | 9.955000 | 0.104500 | 0.596925 | override_hit=0.8 |
| hybrid-v2.json | 200 | 0.735000 | 0.302028 | 5.190000 | 0.581000 | 0.574308 | override_hit=0.7 |
| hybrid.json | 200 | 0.735000 | 0.302028 | 5.190000 | 0.581000 | 0.574308 | override_hit=0.7 |

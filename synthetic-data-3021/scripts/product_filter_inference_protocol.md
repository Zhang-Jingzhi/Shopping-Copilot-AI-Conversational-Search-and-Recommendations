# Product Filter Inference Protocol

## Goal

Infer the simplest plausible organizer-side filtering mechanism that is consistent with 10,187 retained LLO test interactions, 1,406 distinct products, and inclusion of all 200 observed public products.

## Primary metric

Lower is better:

`relative_row_error + relative_product_error + 5 * public_miss_rate`

The three components and the Pareto frontier are always retained separately so an exact count match cannot hide poor public coverage.

## Hypotheses

1. A single monotone row/product threshold generated the pool.
2. Interactions were quality-ranked and capped per product before taking 10,187 rows.
3. Product selection used a weighted mixture of category-normalized popularity, interaction volume, history quality, and metadata quality.
4. No unique rule is identifiable; use consensus across plausible rules to produce high/probable/uncertain/low tiers.

## Read-only inputs

- `data/catalog.jsonl`
- `data/public_set.jsonl`
- `data/upstream/amazon_reviews_2023/5core_llo/Clothing_Shoes_and_Jewelry.test.csv.gz`
- `experiments/results/product_selection_position_3221.csv`

## Outputs

- `experiments/results/product_filter_inference_audit.json`
- `experiments/results/product_filter_inference_3021.csv`
- `experiments/product_filter_inference/results.tsv`

## Constraints

- Do not treat the 3,021 non-public products as negatives.
- Do not claim exact private or 1,406-pool identities without organizer labels.
- Report count fit, public coverage, complexity, and stability separately.
- No external packages or model APIs.

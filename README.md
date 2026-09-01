# Shopping Copilot — Stateful Conversational Product Search

A multi-turn shopping agent for TikTok TechJam 2026, Track 4. It remembers
changing requirements, searches a frozen catalog of 50,000 products, and decides
whether to recommend or ask a clarification question.

**Official entry: `from agent import Agent`.** The submitted configuration runs
offline on CPU using Python's standard library and SQLite FTS5. No GPU, model
weights, Dense embeddings, Qwen/LLM inference, paid API or credentials are needed.
Optional model helper classes in the research code are not enabled by this entry.

## Quick start

Reference interpreter: **CPython 3.12.13**. Python >=3.10 and SQLite with FTS5 are
required; other Python versions have not all been validated. Run from the
repository or extracted submission root:

```bash
python3 -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m submission_tools.verify --code-only
python -m submission_tools.prepare_data
python -m submission_tools.verify
python -m submission_tools.evaluate --offline-check
```

The dependency file intentionally has no third-party packages. Setup downloads
the public organizer kit, verifies its pinned SHA-256, and installs it under
`competition_kit/`. No private team release is needed. Data and labels are not
included in our source ZIP.

If automatic download is unavailable, download `techjam-participant-kit.zip`
from the [official release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit):

```bash
python -m submission_tools.prepare_data --archive /path/to/techjam-participant-kit.zip
```

This offline setup path performs the same verification. Mismatched existing
files are not silently overwritten. Inference needs no network access after setup.

## Official API

```python
from agent import Agent

agent = Agent()
agent.reset("example", {
    "purchase_frequency": "unspecified", "average_prior_rating": None,
    "rating_style": "unspecified", "preference_tags": ["comfort"],
    "summary": "Comfort is a weak preference."
})
response = agent.respond("example", "I need a black dress under $50.", 1, 10)
```

Responses contain `message`, `ask_attribute`, ordered `recommendations` with
catalog `parent_asin` values, and nonnegative token `usage`. The Agent enforces
sequential turns and the ten-turn maximum.
For a judge-provided catalog, pass `Agent(catalog_path=...)` or set
`SHOPPING_CATALOG_PATH`. Default paths are relative to the code, not a developer's
machine. Labels, hidden intent cards, ground truth and future messages are not
Agent inputs. The public evaluation command explicitly uses the verified kit.

## Architecture

```text
User -> Intent -> State -> Pre-retrieval policy (4A)
                            | missing essential context -> Clarify
                            | otherwise -> Retrieval -> Ranking/Post-policy (4B)
                                                        -> Recommend / Clarify
Actual question and shown-product feedback --------------------> State
```

Intent produces explicit set/clear/exclude/remove-exclusion operations. State
accumulates hard constraints, decaying soft preferences, weak profile hints and
question history. Changing a category clears stale product-specific conditions.

The submitted score profile uses two pre-retrieval evidence-collection turns;
4A chooses the question content from current State. From turn three onward,
candidate quality and feedback drive dynamic recommend/clarify decisions. The
resulting State feeds module 2's deterministic keyword, category and
catalog-evidence routes. Its recall-compatible Top-50 contract
protects candidate coverage before the locked CPU reranker returns up to ten.
The alternative `adaptive` profile uses stricter lexical hard checks, dynamic
route weights and variable candidate counts. Textual checks are lexical evidence,
not verified product-variant attributes.

Two warm-up clarification questions and at most one dynamic post-retrieval
question are allowed as a heuristic, distinct from the organizer's ten-turn
limit. One bounded retry may broaden
retrieval depth and remove soft query terms without relaxing hard constraints.
Versioned results reject stale/cross-session handoffs. Actual questions and
shown products are written back into state; errors do not reuse old candidates.

## Results

All 200 official public development sessions, unchanged organizer evaluator:

| Hit@10 | MRR | MTTC | Efficiency | TechnicalScore |
|---:|---:|---:|---:|---:|
| 197/200 (98.5%) | 0.888375 | 3.205 | 0.779500 | 0.914913 |

These are development-set results, not private-test claims. TechnicalScore is
the evaluator's composite, not the overall judging score. A prior fixed-two-
question locked component baseline had MRR 0.884625, the same MTTC 3.205 and
TechnicalScore 0.913788. The submitted integration preserves the full stateful
chain, raises MRR by 0.003750 and raises the composite by 0.001125 on this public
development set.

That component-4 fixed-question baseline and its one-question variant can be
reproduced with `python scripts/reproduce_question_limit_ablation.py`. The two
public-200 outputs and interpretation are documented in the
[question-limit ablation](docs/integration/QUESTION_LIMIT_ABLATION_2026-09-01.md).
This is a separate RankingAgent experiment, not the submitted adaptive pipeline.

Full sample outcomes, code/data hashes, startup, response latency, memory, tokens
and model costs: [public200.json](docs/submission/public200.json).
Clean-package validation: [reproduction.json](docs/submission/reproduction.json).
Interpretation: [technical report](docs/submission/TECHNICAL_REPORT.md).

```bash
python -m submission_tools.evaluate --offline-check
python -m submission_tools.evaluate --sample-id public_0006 --trace --output results/example.json
# Optional strict/adaptive comparison, separate from the submitted default:
python -m submission_tools.evaluate --orchestration-mode adaptive --ranking-mode locked --output results/adaptive-locked.json
# 29 packaged integration and submission tests:
python -m unittest discover -s shopping_agent/tests -v
python -m unittest discover -s submission_tools/tests -v
```

Clarification-policy ablations are explicit and do not change the submitted
default unless selected:

```bash
python -m submission_tools.evaluate --clarification-mode fixed_two_dynamic --output results/fixed-two.json
python -m submission_tools.evaluate --clarification-mode one_then_value --output results/one-then-value.json
```

See the [clarification-policy ablation](docs/integration/CLARIFICATION_POLICY_ABLATION_2026-09-01.md).

The runner calls the original evaluator's `evaluate` function with our Agent;
it does not edit the evaluator, labels, simulator or scoring formula. The offline
check blocks socket connections/DNS in that process. Random session IDs and
timings vary; compare sample-level hits, ranks and turns instead of JSON bytes.

## Recordable demo

```bash
python -m submission_tools.demo --scenario clarify --pause
python -m submission_tools.demo --scenario override --pause
python -m submission_tools.demo --scenario browse --pause
python -m submission_tools.demo --scenario dynamic4b --pause
# Optional: show strict dynamic retrieval and a broad-pool 4B decision
python -m submission_tools.demo --scenario browse --orchestration-mode adaptive --pause
```

The terminal shows actual state, policy decisions, candidate counts, catalog IDs
and product metadata. No front-end is necessary. See the
[recording plan](docs/submission/YOUTUBE_PLAN_ZH.md) and
[Devpost narrative](docs/submission/DEVPOST.md).

## Configuration, cost and limitations

- Default: two state-informed evidence-collection turns followed by dynamic 4B,
  module-2 recall-compatible Top-50 retrieval and locked CPU ranking, with output
  capped at 10.
- `--clarification-mode state_evidence` restores the faster accumulated-evidence
  policy used in the ablation.
- Research comparison: `--orchestration-mode adaptive` enables stricter dynamic
  retrieval and hybrid contextual ranking; it is not the submitted default.
- SQLite and evidence indexes run in memory. Catalog size is about 60.5 MB;
  measured process memory is reported separately and includes the evaluator.
- Model tokens and model API fees are zero. Hardware/electricity costs are not
  estimated. There are no inference credentials or external services.
- `SHOPPING_CATALOG_PATH` overrides the catalog path. No variables are required
  for a normal run. Legacy hooks `TECHJAM_HARD_PENALTY` (0.20) and
  `TECHJAM_POINTWISE_WEIGHT` (0.35; unused without a model) should remain unset
  for reproduction. Effective values are recorded in the report.
- Parsing is rule-based and supports the tested English patterns; some buying
  messages remain unknown. Complex negation, sizing, variant availability and
  missing prices remain limitations. Ranking scores are not probabilities.
- Category matching can admit false positives: a dress request may match a
  dress shirt, and a shoes request may match unrelated items through the broad
  "Clothing, Shoes & Jewelry" taxonomy. State-change demos do not establish
  that every recommended product meets the requested category.
- The provided profile is a weak prior, not persistent cross-session learning.
  The question budget is heuristic. Dense/LLM semantics and cross-category
  diversity are not claimed by this CPU implementation.
- Next steps: controlled clarification/ranking ablations, better attribute
  extraction, and separately validated semantic extensions with CPU fallback.

## Team and attribution

See [contributions](docs/submission/CONTRIBUTIONS.md). Development used Python,
SQLite, Git/GitHub, VS Code, unittest and bdb, with Codex assistance for
integration, debugging and documentation. Measurements come from actual runs.

The organizer's frozen catalog is derived from Amazon Reviews 2023, published
by UCSD McAuley Lab. See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md). Original
data/third-party terms apply; no product images, private artifacts or model
weights are redistributed in our source bundle.

## Build a source-only submission

```bash
python -m submission_tools.build
```

Output: `dist/shopping-copilot-submission.zip` and SHA-256 sidecar. The allowlisted
archive contains source, tests, documents and a per-file manifest, excluding
Git history, catalog/labels, model weights, private documents, caches, virtual
environments and machine-specific debugger configuration. After extraction,
run `python -m submission_tools.verify --code-only`.

# statement_ranking — building the statement-ranking benchmark

Turns `clustering_pipeline/final/` into the user-wise temporal benchmark used by `baselines/` and `LMStaR/`.
Commands are run from `statement_pipeline/`.

```text
prepare   keep interactions with at least one MATCH statement
kcore     iterative user / item 5-core (degree = number of interactions, recomputed to a fixed point)
split     per user, sort by (timestamp, interaction_id): the last interaction is test, the others train
          a test interaction whose item never occurs in train is dropped (not moved to train, no earlier event promoted)
finalize  write the benchmark and check its invariants
```

Settings are in [`configs/pipeline.yaml`](configs/pipeline.yaml) (`k_core.k`, `split.cold_start_items`, output
format). Only final upstream files are read:

```text
<processed_root>/<Dataset>/clustering_pipeline/final/   interactions.jsonl, evidence.jsonl, statements_{user,item,match}.jsonl   (required)
<processed_root>/<Dataset>/evidence_pipeline/final/     reviews_with_statements.jsonl   (optional: normalized statement text;
                                                        otherwise the evidence source_span is used)
```

## Run

```bash
bash statement_ranking/scripts/run_dataset.sh Beauty        # DATASETS_ROOT defaults to ../data/processed
bash statement_ranking/scripts/run_all.sh                   # Beauty, Toys, Yelp (or: run_all.sh Beauty Toys)

# explicit form and resumable stages
python -m statement_ranking.run --dataset Beauty --dataset-root ../data/processed/Beauty \
    [--output-root DIR] [--stage prepare|kcore|split|finalize|all | --from-stage S] [--to-stage S] [--force]
```

Intermediate results are in `statement_ranking/{00_prepare,01_kcore,02_split}/`; a rerun skips completed stages.

## Output (`<processed_root>/<Dataset>/statement_ranking/final/`)

| file | content |
|---|---|
| `interactions_{user,item,match}.csv` | one row per interaction: `interaction_id, line_index, user_id, item_id, rating, timestamp, split`, user / item / global sequence indices, `statement_ids`, `aligned_statement_ids`, `original_statement_ids` |
| `interactions.csv` | the three views in one table (`{user,item,match}_statement_ids`, …) |
| `statements_{type}.csv`, `id2statement_{type}.json` | canonical statements (cluster IDs) |
| `original_statements_{type}.csv`, `id2original_{type}.json` | statement occurrences: `original_statement_id` (evidence ID), canonical ID, interaction, `source_span`, `text`, `stance` |
| `item_candidates_{type}.csv` | per item, the union of its statements over the retained interactions (the candidate set $C_i$ uses `match`) |
| `train.index`, `test.index` | interaction IDs of each split |
| `users.csv`, `items.csv`, `excluded_cold_start.csv` | entities; test events removed by the cold-start rule |
| `manifest.json`, `audit.json` | statistics, configuration and invariant checks |

ID lists are `:`-separated. `statement_ids` is the ordered set of canonical IDs of an interaction;
`aligned_statement_ids` is aligned one-to-one with `original_statement_ids` (one per extracted occurrence).

`audit.json` checks, and finalization fails otherwise, that every interaction has MATCH statements, the 5-core
holds before the split, each user has at most one test event and it is their latest, no test item is cold-start,
aligned and original ID lists have equal length and deduplicate to `statement_ids`, and every referenced ID exists.

The benchmark stores no static profile: sequence indices let models build each user's and item's history from
the interactions before the target one (as LMStaR and the baselines do).

## Loader

```python
from statement_ranking.loader import StatementRankingDataset

ds = StatementRankingDataset("../data/processed/Beauty/statement_ranking")
for it in ds.interactions("match", split="test"):
    print(it.user_id, it.item_id, it.statement_ids)
id2match = ds.id2statement("match")
candidates = ds.item_candidates("match")
train_ids, test_ids = ds.train_index(), ds.test_index()
```

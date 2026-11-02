# statement_pipeline — from raw reviews to statement-ranking datasets

Builds the statement-ranking benchmark of the paper in four stages. It refines the StaR construction: the extraction
prompt targets statements that explain the match between a user's preferences and an item, and cluster merges are
verified by a reranker on several representatives of each cluster.

```text
process            raw Amazon / Yelp files  -> reviews.jsonl, meta_items.jsonl, meta_users.jsonl
evidence_pipeline  LLM extraction of atomic statements (user / item / interaction), projection, stance
clustering_pipeline paraphrase clustering of USER / ITEM / MATCH statements (embedding kNN + reranker)
statement_ranking  MATCH interactions, 5-core, user-wise temporal split, benchmark export
```

All commands are run **from this folder** (`cd statement_pipeline`). Paths come from
[`configs/paths.yaml`](configs/paths.yaml): `data_root` (raw files, default `../data/raw`) and `processed_root`
(everything generated, default `../data/processed`); `${VAR}` and `~` are expanded.

| stage | README | output (under `processed_root/<Dataset>/`) |
|---|---|---|
| process | below | `reviews.jsonl`, `meta_items.jsonl`, `meta_users.jsonl`, `stats.json` |
| evidence | [evidence_pipeline](evidence_pipeline/README.md) | `evidence_pipeline/final/reviews_with_statements.jsonl` |
| clustering | [clustering_pipeline](clustering_pipeline/README.md) | `clustering_pipeline/final/` |
| benchmark | [statement_ranking](statement_ranking/README.md) | `statement_ranking/final/` (used by `baselines/` and `LMStaR/`) |

## Installation

```bash
bash evidence_pipeline/scripts/create_env.sh [statementrec]    # conda env: Python 3.11, vLLM, PyYAML, tqdm, torch, transformers
# CPU-only stages (process, statement_ranking) only need: pip install pyyaml tqdm
```

`create_env.sh [ENV_NAME] [CUDA_BACKEND] [PROJECT_ROOT]` installs vLLM for the given CUDA backend (default `cu129`)
and checks structured outputs. `evidence_pipeline/scripts/check_gpu_env.sh` and
`evidence_pipeline/scripts/check_hf_access.sh MODEL` verify the GPU environment and access to a gated model.

## Stage 1 — process the raw files

[`configs/datasets.yaml`](configs/datasets.yaml) lists, per dataset, its kind (`amazon2014`, `yelp2013`) and the
review / item / user files relative to `data_root`:

```text
data/raw/AmazonReviews2014/Beauty/{reviews_Beauty_5.json, meta_Beauty.json}
data/raw/AmazonReviews2014/Toys/{reviews_Toys_and_Games_5.json, meta_Toys_and_Games.json}
data/raw/Yelp2013/{review_Yelp.json, meta_business_Yelp.json, meta_user_Yelp.json}
```

```bash
python -m process.process_datasets --all --overwrite          # or: --dataset Beauty [--dataset Toys]
bash process/scripts/process_all.sh                           # same as --all --overwrite
```

Every dataset is written to `processed_root/<Dataset>/`: `reviews.jsonl` (`user_id, item_id, timestamp, rating,
review_title, review`), `meta_items.jsonl`, `meta_users.jsonl` (derived from the reviews for Amazon, which has no user file)
and `stats.json`; `processed_root/processing_summary.json` collects the statistics. Options are in
[`configs/processing.yaml`](configs/processing.yaml) (keep only referenced metadata, drop empty fields).

## Stages 2–4, end to end (one dataset)

```bash
# 2. extraction: 3 stages + recovery + composition (SLURM; see the evidence README for local runs)
python -m evidence_pipeline.slurm.submit_pipeline --dataset Beauty
#    publish the composed run as the input of the next stages
cd ../data/processed/Beauty/evidence_pipeline && ln -sfn combined/<stage1_lineage>/<stages2-3_lineage> final && cd -
# 3. clustering (resumable; one GPU)
bash clustering_pipeline/scripts/run_dataset.sh Beauty
# 4. benchmark
bash statement_ranking/scripts/run_dataset.sh Beauty
```

`clustering_pipeline/scripts/run_dataset.sh` and `statement_ranking/scripts/run_dataset.sh` read the datasets from
`DATASETS_ROOT` (default `../data/processed`).

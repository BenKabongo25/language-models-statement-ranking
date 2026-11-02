# clustering_pipeline — paraphrase clustering of statements

Clusters the USER, ITEM and MATCH statements of `evidence_pipeline/final/reviews_with_statements.jsonl` separately,
so that paraphrases share one canonical statement. Commands are run from `statement_pipeline/`.

```text
prepare -> embed -> neighbors -> rerank -> cluster -> finalize -> audit
```

| stage | what it does | default |
|---|---|---|
| `prepare` | collects the statements of `ok` records, by type (`interaction` is clustered as `match`) | |
| `embed` | embeds every statement | Qwen3-Embedding-0.6B, last-token pooling, L2-normalized |
| `neighbors` | k-nearest neighbors above a cosine threshold (FAISS if installed, else torch) | k = 128, cosine ≥ 0.80 |
| `rerank` | scores every candidate pair with a type-specific equivalence instruction | Qwen3-Reranker-0.6B |
| `cluster` | Sentinel Frontier Agglomeration (below) | threshold 0.95, 4 sentinels |
| `finalize` | canonical statements, interactions, evidence, item candidates → `final/` | |
| `audit` | pair-level audit table (`05_pair_analysis/pairs_<type>.tsv`) | |

## Clustering rule

Each cluster is represented by 4 sentinels: its cosine medoid, then farthest-first points. Two clusters merge only
if **every** cross-sentinel pair (at most 4 × 4 = 16) gets a reranker score ≥ 0.95, and any observed score below the
threshold is a permanent veto. Reranker-positive seed edges are sorted once by score and consumed once; merge
proposals that conflict with another merge of the same GPU wave go to a retry frontier and are re-validated against
the new sentinels. Clustering stops when the seed stream is exhausted and the frontier is empty (a fixed point, no
global rounds). Missing sentinel comparisons of a wave are deduplicated and scored in large batches. Finalization
checks that no reranker-observed pair below the threshold lies inside a cluster.

All parameters are in [`configs/pipeline.yaml`](configs/pipeline.yaml). Parameters marked *execution-only* (batch
sizes, index type, checkpoint frequency) do not invalidate cached artifacts; `cluster.max_merge_attempts` /
`max_new_reranker_pairs` are optional budgets (the stage checkpoints and exits when reached).

## Run

```bash
pip install -r clustering_pipeline/requirements.txt      # + faiss-gpu / faiss-cpu (optional, faster neighbors)
bash clustering_pipeline/scripts/run_dataset.sh Beauty                       # all stages, resumable
bash clustering_pipeline/scripts/run_dataset.sh Beauty --from-stage cluster  # recompute cluster and downstream
bash clustering_pipeline/scripts/run_dataset.sh Beauty --from-stage cluster --to-stage cluster
bash clustering_pipeline/scripts/run_dataset.sh Beauty --from-stage cluster --fresh-cluster   # ignore checkpoints
```

`run_dataset.sh DATASET [args]` calls
`python -m clustering_pipeline.run --input $DATASETS_ROOT/DATASET/evidence_pipeline/final/reviews_with_statements.jsonl --output-root $DATASETS_ROOT/DATASET/clustering_pipeline`
(`DATASETS_ROOT` defaults to `../data/processed`). Other options: `--stage <name>|all|doctor` (run up to a stage),
`--types user item match`, `--device auto|cpu|cuda`, `--force`, `--audit-ranks auto|available|recompute`.

Every stage is resumable: rerunning continues from the last checkpoint (`04_clusters/state_<type>/`: partition, seed
cursor, retry frontier, statistics). Without `--fresh-cluster`, the cluster stage also resumes from an earlier
partition of a previous implementation (merge trace or `raw_to_cluster_<type>.npz`), reusing all cached reranker
scores; the manifest records the initialization source.

## Outputs (`<processed_root>/<Dataset>/clustering_pipeline/`)

```text
00_prepare/ … 03_rerank/       intermediate artifacts (embeddings, neighbor and reranker shards)
04_clusters/                   clusters_<type>.jsonl, raw_to_cluster_<type>.npz, merge_trace_<type>.tsv,
                               frontier_progress_<type>.tsv (one row per GPU wave), state_<type>/, verification_<type>/
05_pair_analysis/              pairs_<type>.tsv: cosine, neighbor ranks, reranker score, final clusters
final/                         interactions.jsonl, evidence.jsonl, statements_{user,item,match}.jsonl (+ manifests)
```

Only `final/` is needed by `statement_ranking`.

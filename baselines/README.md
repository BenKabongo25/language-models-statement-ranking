# baselines — explanation-ranking baselines on the statement-ranking benchmark

All baselines of the paper behind one item-level training / evaluation protocol, keeping each method's objective
and initialization from its paper or official code. Commands are run from the **repository root**. The only input
is the benchmark built by `statement_pipeline/statement_ranking`:

```text
data/processed/<Dataset>/statement_ranking/final/{interactions_match.csv, id2statement_match.json, item_candidates_match.csv}
```

## Methods

| `--model` | method | source | seeds |
|---|---|---|---|
| `rand` | RAND | EXTRA / StaR | 3 |
| `rucf`, `ricf` | user- / item-based collaborative filtering | BPER, EXTRA | 1 |
| `userpop`, `itempop`, `globalpop` | popularity | Rank, Don't Generate (StaR) | 1 |
| `cd` | Canonical Decomposition | BPER paper | 3 |
| `pitf` | PITF | BPER paper | 3 |
| `bper` | BPER | Li, Zhang & Chen, `lileipisces/BPER` | 3 |
| `bper_plus` | BPER+ (BERT) | `lileipisces/BPER` | 3 |
| `bper_large` | BPER-large (768-d, random init.) | SE-BPER paper | 3 |
| `se_bper`, `se_bper_plus` | SE-BPER, SE-BPER+ (ALBERT) | `zhichaoxu-shufe/Semantic-Enhanced-BPER` | 3 |
| `expgcn` | ExpGCN | `Joinn99/ExpGCN` | 3 |
| `cler` | CLER | `steip/CLER` | 3 |

## Protocol

- **Split.** The benchmark split is kept: each user's `test` row is their last interaction. For tuning, the latest
  remaining `train` row of every user with a test row is held out as validation; users without a test row stay in
  training.
- **Learned methods.** Hyperparameters are selected on train → validation (`baselines.tune`), frozen, then the model
  is refit on train + validation and tested once per seed (40, 41, 42). Per-seed selection is rejected by the runner.
  Deterministic methods (`rucf`, `ricf`, `*pop`) are fitted once on train + validation.
- **Candidates.** For a target `(u, i)` every method ranks the full candidate set $C_i$ of item `i`
  (`item_candidates_match.csv`), the same for all methods.
- **Negatives.** Every explanation-ranking loss samples negatives among the statements of the target item observed
  in the fit split, excluding all positives of the same `(u, i)` (`sampling: {scope: item, candidate_source:
  fit_only}`; `candidate_source: all` is a leakage ablation). ExpGCN's auxiliary item-recommendation loss keeps its
  original negatives.
- **Metrics.** One shared implementation of binary Precision, Recall, F1, NDCG and HR at K ∈ {5, 10}; NDCG uses an
  IDCG truncated to `min(K, |P_ui|)`. Per-interaction metrics and top-K IDs are saved for paired tests.

### Fidelity notes

- **BPER**: two BPR terms and $r(u,i,e)=\mu\,r_U(u,e)+(1-\mu)\,r_I(i,e)$, explanation-bias initialization of the
  NumPy code, one positive per interaction and epoch; vectorized in PyTorch with L2 on sampled rows.
- **BPER+**: BERT `[CLS]` → linear projection ⊙ explanation factor. The bias broadcast of the official code is kept
  (`official_legacy_bias_broadcast: true`); `configs/regimes/bper_plus_paper_consistent.yaml` follows the equations.
- **BPER-large / SE-BPER / SE-BPER+**: 768-d family of the SE-BPER code: fixed `num_negatives` per positive,
  ALBERT representations divided by 100, biases in [−0.01, 0.01], and the official Adam branch that ignores weight
  decay (`official_adam_ignores_weight_decay: true`; `configs/regimes/bper_family_fair.yaml` makes L2 operative). SE-BPER+
  first fine-tunes `albert-base-v2` with the triplet-context objective, recomputing profiles every epoch.
- **ExpGCN**: U–I, U–statement and I–statement graphs, symmetric normalization, layer-mean aggregation, multi-task
  BPR; the U–I branch is initialized with ID factors plus explanation-branch representations, as in the official code.
- **CLER**: ID embeddings, normalized weighted user–item query, temperature, balanced NT-BXent; its global in-batch
  negatives are replaced by same-item negatives.

## Configuration

`configs/base.yaml` (data and output roots, split, sampling, metrics) ← `configs/models/<model>.yaml` ←
`--extra-config` files (merged in this order) ← `--set key.sub=value`. `configs/regimes/` holds search spaces
(`bper_mu_search`, `bper_plus_lr_search`, `tf_original_search`, `validation_mu`) and controlled variants
(`bper_family_fair`, `bper_plus_paper_consistent`).

## Run

```bash
pip install -r baselines/requirements.txt
pytest -q baselines/tests

# one run (one method × one dataset × one seed)
python -m baselines.run --dataset Beauty --model bper --seed 40 [--extra-config FILE] [--set evaluation.save_full_ranking=true]

# validation tuning, then the frozen selection for every test seed
python -m baselines.tune --dataset Beauty --model bper --search-config baselines/configs/regimes/bper_mu_search.yaml
python -m baselines.run --dataset Beauty --model bper --seed 40 \
    --extra-config data/processed/Beauty/baselines/bper/tuning/selected_overrides.yaml

# mean, sample std and per-seed values
python -m baselines.aggregate --dataset Beauty --model bper
bash baselines/scripts/aggregate_all.sh                      # all methods, Beauty / Toys / Yelp
```

`bper_plus`, `se_bper` and `se_bper_plus` download `bert-base-uncased` / `albert-base-v2`; on nodes without internet,
prefetch them first: `python -m baselines.prefetch --config baselines/configs/base.yaml --config baselines/configs/models/se_bper.yaml`.

### SLURM

```bash
python -m baselines.slurm.submit [--datasets Beauty Toys Yelp] [--models ...] [--seeds 40 41 42] \
    [--profile auto|baselines/configs/slurm/<profile>.yaml] [--extra-config FILE] [--dry-run]
bash baselines/scripts/submit_all.sh            # same, all defaults
python -m baselines.slurm.submit_tuning --datasets Beauty --models bper \
    --search-config baselines/configs/regimes/bper_mu_search.yaml
```

One job = one method × dataset × seed (deterministic methods: one job, `seed_0`). `--profile auto` uses
`configs/slurm/cpu.yaml` for deterministic methods and RAND, `h200.yaml` for the Transformer methods and ExpGCN, and
`a6000.yaml` otherwise. These are example profiles: set `partition`, resources and `conda_env` for your cluster.
Jobs run `slurm/run_one.sbatch` (activates `conda_env`, prefetches Transformer weights, logs to `baselines/logs/`).

## Outputs

```text
data/processed/<Dataset>/baselines/<model>/
├── seed_<seed>/   config_resolved.yaml, best_config.yaml, split_stats.json, fit_info.json, metrics.json,
│                  per_interaction.csv, full_rankings.jsonl (if evaluation.save_full_ranking)
├── tuning/        tuning_results.json, selected_overrides.yaml, selected_full_config.yaml
└── aggregate.json, aggregate.csv
```

## References

- Li, Zhang, Chen. *On the Relationship between Explanation and Recommendation: Learning to Rank Explanations for Improved Performance.* — [lileipisces/BPER](https://github.com/lileipisces/BPER)
- Xu et al. *Learning to Rank Rationales for Explainable Recommendation.* — [zhichaoxu-shufe/Semantic-Enhanced-BPER](https://github.com/zhichaoxu-shufe/Semantic-Enhanced-BPER)
- Wei et al. *ExpGCN: Review-aware Graph Convolution Network for Explainable Recommendation.* — [Joinn99/ExpGCN](https://github.com/Joinn99/ExpGCN)
- Escarda-Fernández et al. *Contrastive Learning for Explanation Ranking.* — [steip/CLER](https://github.com/steip/CLER)
- Kabongo et al. *Rank, Don't Generate: Statement-level Ranking for Explainable Recommendation.*

# Learning to Rank Explanations for Recommendation with Language Models

Code and data pipeline of **LMStaR** (Language Models for Statement Ranking), a fully text-based framework that
formulates personalized explanation ranking as dense retrieval: a query encoder represents the user's interaction
history and the target item, a statement encoder represents each candidate explanation, and both are adapted jointly
with a multi-positive contrastive objective. The repository also contains the pipeline that builds the
statement-ranking datasets from raw reviews, and the baselines of the paper.

## Repository layout

```text
.
├── statement_pipeline/      # raw reviews -> statement-ranking datasets
│   ├── process/             #   1. harmonize raw Amazon / Yelp files
│   ├── evidence_pipeline/   #   2. LLM extraction of explanatory statements (vLLM)
│   ├── clustering_pipeline/ #   3. paraphrase clustering of the statements (embedding + reranker)
│   └── statement_ranking/   #   4. filtering, k-core, temporal split, benchmark export
├── baselines/               # popularity, collaborative, factorization, BPER family, ExpGCN, CLER
└── LMStaR/                  # the LMStaR dual encoder: training, evaluation, transfer, zero-shot
```

Each folder has its own README with every command:
[`statement_pipeline`](statement_pipeline/README.md) ·
[`baselines`](baselines/README.md) ·
[`LMStaR`](LMStaR/README.md).

## Data flow

```text
data/raw/<Dataset> files (reviews, item / user metadata)
   │ statement_pipeline/process            -> data/processed/<Dataset>/{reviews,meta_items,meta_users}.jsonl
   │ statement_pipeline/evidence_pipeline  -> data/processed/<Dataset>/evidence_pipeline/final/
   │ statement_pipeline/clustering_pipeline-> data/processed/<Dataset>/clustering_pipeline/final/
   │ statement_pipeline/statement_ranking  -> data/processed/<Dataset>/statement_ranking/final/   (the benchmark)
   ├── baselines                           -> data/processed/<Dataset>/baselines/<method>/seed_<seed>/
   └── LMStaR                              -> data/processed/LMStaR/<Dataset>/runs/<backbone>/lmstar/<run>/
```

All components share one convention: original files in `data/raw/`, everything generated in `data/processed/`, both
at the repository root (ignored by git). To use existing storage, link it:
`ln -s /path/to/raw data/raw && ln -s /path/to/processed data/processed`, or change the roots in the configs
(`statement_pipeline/configs/paths.yaml`, `baselines/configs/base.yaml`, `LMStaR/configs/base.yaml`).

## Datasets

| | Beauty | Toys | Yelp |
|---|---:|---:|---:|
| Users | 10,819 | 8,556 | 5,209 |
| Items | 6,289 | 5,662 | 3,787 |
| Interactions | 97,484 | 74,775 | 83,298 |
| Unique statements | 59,803 | 63,333 | 81,516 |

Sources: [Amazon Reviews 2014](https://jmcauley.ucsd.edu/data/amazon/links.html) (5-core *Beauty* and
*Toys and Games* reviews + metadata) and the [Yelp dataset](https://www.yelp.com/dataset/challenge); the expected
file names are listed in `statement_pipeline/configs/datasets.yaml`. The baselines and LMStaR only need the
benchmark files (`data/processed/<Dataset>/statement_ranking/final/`): with them, start directly at step 5 below.

## Reproducing the paper

| step | where | hardware |
|---|---|---|
| 1. process raw files | `statement_pipeline/` | CPU |
| 2. extract statements (Qwen3.8-27B, Qwen3.5-9B) | `statement_pipeline/evidence_pipeline` | GPU, vLLM (SLURM launchers provided) |
| 3. cluster paraphrases (Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B) | `statement_pipeline/clustering_pipeline` | 1 GPU |
| 4. build the benchmark (5-core, user-wise temporal split) | `statement_pipeline/statement_ranking` | CPU |
| 5. baselines (3 seeds) | `baselines/` | CPU / 1 GPU |
| 6. LMStaR (5 backbones, LoRA / full fine-tuning, transfer, zero-shot) | `LMStaR/` | 1–2 GPUs (H200 for 4B / 8B) |

Minimal path from the released benchmark, from the repository root:

```bash
pip install -r baselines/requirements.txt
python -m baselines.run --dataset Beauty --model cler --seed 40

cd LMStaR && pip install -e .
python -m lmstar.train --config configs/base.yaml --config configs/models/qwen3-embedding-0.6b.yaml \
    --config configs/datasets/beauty.yaml --seed 40
```

Each component has its own dependencies (the extraction stage needs vLLM, LMStaR needs PEFT); separate Python
environments are recommended. Gated Hugging Face models (Gemma, EmbeddingGemma, Llama) need `export HF_TOKEN=...`
and their licence accepted on the Hub.

## Citation

```bibtex
@article{kabongo2026lmstar,
  title   = {Learning to Rank Explanations for Recommendation with Language Models},
  author  = {Kabongo, Ben and Guigue, Vincent},
  year    = {2026}
}
```

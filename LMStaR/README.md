# LMStaR — Learning to Rank Explanations for Recommendation with Language Models

LMStaR ranks the candidate explanation statements of a target item (*"smells like vanilla"*, *"great value for the
price"*) for a user, from the user's interaction history. It is a dual encoder on any Hugging Face backbone
(embedding models, LLMs, BERT-like encoders), adapted with LoRA (default) or full fine-tuning:

$$q_x=\mathrm{pool}\big(f_\theta(\text{instruction}\,\|\,\text{history}\,\|\,\text{target item})\big),\qquad
d_s=\mathrm{pool}\big(g_\phi(s)\big),\qquad \mathrm{score}(x,s)=\langle q_x,d_s\rangle$$

The query tower encodes the history (the statements of past interactions, newest first, packed into the token
budget) and the target item's metadata; the statement tower encodes each statement. Both use EOS (default), mean or
CLS pooling with L2 normalization and are trained with a multi-positive InfoNCE loss against negatives drawn among
the item's statements. Evaluation ranks **all** candidate statements of the item. Data-parallel training is exact
(gradient cache over both towers, world-size-invariant sampling). Commands are run from this folder (`cd LMStaR`).

## Installation

```bash
pip install -e .        # torch >= 2.2, transformers >= 4.51, peft, pyyaml, tensorboard, safetensors
```

Gated backbones (Llama, Gemma, EmbeddingGemma) need `export HF_TOKEN=...` and their licence accepted on the Hub.

## Data and protocol

LMStaR reads the benchmark of `statement_pipeline/statement_ranking` from
`<dataset.root>/<Dataset>/statement_ranking/final/` (`dataset.root` defaults to `../data/processed`):
`interactions.csv` (split, user sequence index, `match_statement_ids`), `items.csv` (item metadata),
`id2statement_match.json` and `item_candidates_match.csv` (the candidates ranked at evaluation).

The last `train` interaction of every test user is held out for validation; FIT = train − validation. Training
histories use the FIT interactions strictly before the target, validation histories all of FIT, and test histories
FIT + validation (`evaluation.test_context: fit_plus_val`; `fit_only` excludes the validation interaction).
Negatives are the item's FIT statements minus the user–item positives. No evaluation example is dropped; the best
checkpoint is selected on validation NDCG@5.

## Configuration

Configs merge left to right: [`configs/base.yaml`](configs/base.yaml) (all defaults) ← `models/<m>.yaml` ←
`datasets/<d>.yaml` ← `hardware/` ← `variants/` ← `init/` ← `--set section.key=value` (unknown keys are rejected).

| folder | content |
|---|---|
| `models/` | backbone id, native prompts, maximum length, per-GPU micro-batches: `qwen3-embedding-{0.6b,4b,8b}`, `f2llm-4b`, `gte-qwen2-1.5b`, `qwen3-{0.6b,1.7b,4b}`, `llama-3.2-{1b,3b}`, `gemma-3-1b`, `smollm2-1.7b`, `phi-4-mini`, `embeddinggemma-300m`, `bge-{m3,large,base}`, `e5-large-v2`, `multilingual-e5-large-instruct`, `arctic-embed-l-v2`, `gte-modernbert-base`, `modernbert-{embed-base,base,large}`, `nomic-embed-text-v1.5`, `mxbai-embed-large-v1`, `blair-roberta-{base,large}`, `contriever-msmarco` |
| `datasets/` | `beauty`, `toys`, `yelp` (paper), `sports` |
| `hardware/` | `1gpu_h200` (64 × 4), `2gpu_h200` (7–8B backbones): same effective batch of 256 as the default 128 × 2 |
| `variants/` | `full_finetuning`, `shared_tower`, `query_only`, `pooling_{mean,cls}`, `negatives_userpop{25,50}`, `history_{chronological,random,random_order,skip,cap3,target_first}`, `test_fit_only` |
| `init/` | `from_checkpoint`: start from a trained run (domain adaptation) |

Default (paper) settings: LoRA r = 16, α = 32 on the attention projections of both towers, learning rate 1e-4
(2e-5 for full fine-tuning), temperature 0.1, 32 negatives per positive, batch 128 × 2 accumulation steps,
10 epochs with early stopping (patience 5 evaluations), query length 4,096 tokens, statement length 256.
`training.batch_size` is the global batch: `--gpus N` splits it without changing the computation.

## Training

```bash
C="--config configs/base.yaml --config configs/models/qwen3-embedding-0.6b.yaml --config configs/datasets/beauty.yaml"
python -m lmstar.train $C --seed 40                                            # 1 GPU
python -m lmstar.train $C --config configs/hardware/1gpu_h200.yaml --seed 40   # backbones >= 1.5B
CUDA_VISIBLE_DEVICES=0,1 python -m lmstar.train --config configs/base.yaml \
    --config configs/models/qwen3-embedding-8b.yaml --config configs/datasets/yelp.yaml \
    --config configs/hardware/2gpu_h200.yaml --gpus 2 --seed 40                # 8B, 2 GPUs

python -m lmstar.train $C --config configs/variants/full_finetuning.yaml       # no LoRA (fp32 weights, bf16 autocast)
python -m lmstar.train $C --config configs/variants/shared_tower.yaml          # one tower for queries and statements
python -m lmstar.train $C --config configs/variants/query_only.yaml            # frozen statement encoder
python -m lmstar.train $C --set model.query_max_length=2048 --set training.temperature=0.05
python -m lmstar.train --config configs/base.yaml --config configs/models/qwen3-embedding-0.6b.yaml \
    --config configs/datasets/toys.yaml --config configs/init/from_checkpoint.yaml \
    --set training.init_checkpoint=<beauty_run>/best                           # transfer with adaptation
```

A run is written to `<output.root>/<Dataset>/runs/<backbone>/lmstar/<run_id>/` (`output.root` defaults to
`../data/processed/LMStaR`): `config.yaml`, `train.log`, TensorBoard events, `best/` (LoRA adapters
`{query,statement}_lora`, or `*_model` for full fine-tuning, `shared_*` for a shared tower, tokenizer,
`metadata.json`), `test_metrics.json` and `test_predictions.csv` (one row per candidate: score, label, rank).
Tokenized statements are cached in `<output.root>/<Dataset>/token_cache/`.

## Evaluation

```bash
python -m lmstar.evaluate --runs <run_a> <run_b> --splits val test [--gpus 2]                  # re-evaluation
python -m lmstar.evaluate --dataset Beauty --backbone Qwen3-Embedding-8B --select top:1 \
    --filter L4096_ --splits test                                                           # discover runs by id (--list)
python -m lmstar.evaluate --runs <beauty_run> --target-dataset Toys --target-dataset Yelp      # zero-shot transfer
python -m lmstar.evaluate --zero-shot $C --splits test                                         # backbone without training
python -m lmstar.evaluate --runs <run> --test-context fit_only --tag fit_only                  # other test context
```

Results (`metrics.json`, `test_predictions.csv`, `eval.log`) are written to `<run>/reeval/<tag>/`,
`<run>/transfer/<Target>/<tag>/` and `<output.root>/<Dataset>/zero_shot/<model>/<tag>/`, and collected in
`<output.root>/{reeval,transfer,zero_shot}_<tag>.csv`. Metrics: NDCG, Precision, Recall, F1 and HR at 5 and 10 (× 100).

## Code

```text
lmstar/
├── config.py         defaults, YAML composition, --set
├── data/             dataset, history packing and prompts, token cache, padding
├── model/            encoder towers (LoRA / full, pooling), tokenizer setup
├── training/         InfoNCE objective, negative sampling, gradient cache
├── train.py          training loop and entry point (single or multi-GPU)
├── evaluate.py       re-evaluation, transfer, zero-shot
├── inference.py      scoring of all candidates, loading of trained towers
└── metrics.py        ranking metrics
```

## Citation

```bibtex
@article{kabongo2026lmstar,
  title   = {Learning to Rank Explanations for Recommendation with Language Models},
  author  = {Kabongo, Ben and Guigue, Vincent},
  year    = {2026}
}
```

# evidence_pipeline — LLM extraction of explanatory statements

Extracts atomic statements from every review with vLLM, in three stages, each followed by a targeted recovery pass,
then composes the final file read by the clustering stage. Commands are run from `statement_pipeline/`.

```text
Stage 1 primary -> Stage 1 recovery -> resolved Stage 1
                                        ├─> Stage 2 primary -> Stage 2 recovery ─┐
                                        └─> Stage 3 primary -> Stage 3 recovery ─┴─> compose
```

| stage | input | output per unit | default model |
|---|---|---|---|
| 1 — atomic evidence | review + item title (the rating is not shown) | `{source_span, statement, type: user / item / interaction / external}` | Qwen3.8-27B |
| 2 — interaction projection | one Stage-1 `interaction` + its `source_span` | `{user, item}` (the interaction itself is the `match` statement) | Qwen3.5-9B |
| 3 — stance | one Stage-1 `user` or `interaction` + its `source_span` | `{stance: positive / negative / neutral}` (`item` / `external`: `null`) | Qwen3.5-9B |

Prompts are in [`prompts/`](prompts); stage settings (batch sizes, retries, sampling, prompt revisions) in
[`configs/pipeline.yaml`](configs/pipeline.yaml); models in [`configs/models/`](configs/models)
(`qwen38_27b`, `qwen35_9b`, `gemma3_27b`: vLLM settings, `requires_hf_token`). Generation is greedy and
schema-constrained (structured outputs); outputs are validated against [`schemas.py`](schemas.py).

## Recovery

Each primary pass is deterministic and non-thinking. Recovery reruns, with **the same prompt** (its SHA256 is
checked), every record whose status is `error` or `partial` or whose payload is empty (Stage 1 without statements,
Stage 2 without projection, Stage 3 without stance), with its own sampling profile (non-zero temperature, more
retries and tokens, thinking when the model family supports it; Gemma 3 always runs non-thinking). Recovery never
edits the primary `results.jsonl`: it writes `resolved_results.jsonl`, which keeps the better record per key (a
non-empty partial Stage-1 extraction is never replaced by an empty one). Its `manifest.json` reports
`target_count`, `selected_recovery_count`, `unresolved_target_count` and the resolved `ok / partial / error` counts.

## Outputs

```text
<processed_root>/<Dataset>/evidence_pipeline/
├── stage1/<run>/                       manifest.json, results.jsonl, shards/, recovery/<recovery_run>/resolved_results.jsonl
├── stage2/<stage1 lineage>/<run>/       same structure
├── stage3/<stage1 lineage>/<run>/       same structure
├── combined/<stage1 lineage>/<stage 2-3 lineage>/reviews_with_statements.jsonl
└── final -> combined/<...>/<...>        link to the composed run used downstream (created by you)
```

Run names are `<model>__<prompt revision>` (e.g. `qwen3.8-27b__atomic-v1`, with `__limit<N>` for a `--limit` run)
and `<model>__failure-recovery-v1__thinking` for recoveries. Every manifest stores the SHA256 of the exact parent
file it consumed, and composition rejects inconsistent lineages.

## Run with SLURM

[`configs/slurm/`](configs/slurm) holds example profiles (`h200.yaml`, `a6000.yaml`): set `partition` (and
`account`, `qos`, `nodelist` if needed), resources and `conda_env` for your cluster. Logs go to
`evidence_pipeline/logs/`.

```bash
# whole pipeline with SLURM dependencies: 3 primary passes, 3 recoveries, composition
python -m evidence_pipeline.slurm.submit_pipeline --dataset Beauty [--limit 50] [--dry-run]

# or stage by stage
python -m evidence_pipeline.slurm.submit_stage --stage 1 --dataset Beauty \
    --model-config evidence_pipeline/configs/models/qwen38_27b.yaml \
    --profile evidence_pipeline/configs/slurm/h200.yaml --gpus 1
python -m evidence_pipeline.slurm.submit_recovery --stage 1 --dataset Beauty --run-name qwen3.8-27b__atomic-v1 \
    --model-config evidence_pipeline/configs/models/qwen38_27b.yaml \
    --profile evidence_pipeline/configs/slurm/h200.yaml --gpus 1
python -m evidence_pipeline.slurm.submit_stage --stage 2 --dataset Beauty \
    --stage1-run qwen3.8-27b__atomic-v1 --stage1-recovery-run qwen3.8-27b__failure-recovery-v1__thinking \
    --model-config evidence_pipeline/configs/models/qwen35_9b.yaml \
    --profile evidence_pipeline/configs/slurm/a6000.yaml             # same for --stage 3, then their recoveries
```

`submit_pipeline` takes `--stageN-model`, `--stageN-profile`, `--stageN-gpus` and their `--stageN-recovery-*`
counterparts; `submit_recovery` can override `--temperature`, `--max-tokens`, `--thinking-token-budget`, `--retries`.
Shortcuts: `bash evidence_pipeline/scripts/submit_beauty_pilot.sh [N]` (Beauty, first N reviews),
`bash evidence_pipeline/scripts/submit_stage1_recovery.sh DATASET [RUN]`,
`bash evidence_pipeline/scripts/submit_all_stage1_recoveries.sh [RUN]` (Beauty, Toys, Yelp).

## Run without SLURM

The same steps with the worker / merge commands (one shard here; `--num-shards N --shard-index i` splits a pass
over N processes):

```bash
M1=evidence_pipeline/configs/models/qwen38_27b.yaml; M2=evidence_pipeline/configs/models/qwen35_9b.yaml
S1=qwen3.8-27b__atomic-v1; R1=qwen3.8-27b__failure-recovery-v1__thinking

python -m evidence_pipeline.run_stage --stage 1 --dataset Beauty --model-config $M1      # [--limit N | --indices-file F]
python -m evidence_pipeline.merge --stage 1 --dataset Beauty --run-name $S1
python -m evidence_pipeline.run_recovery --stage 1 --dataset Beauty --model-config $M1 --run-name $S1
python -m evidence_pipeline.merge_recovery --stage 1 --dataset Beauty --run-name $S1 --recovery-run-name $R1

for STAGE in 2 3; do                     # stage 2: projection-v1, stage 3: stance-v1
  REV=$([ $STAGE = 2 ] && echo projection-v1 || echo stance-v1)
  python -m evidence_pipeline.run_stage --stage $STAGE --dataset Beauty --model-config $M2 \
      --stage1-run $S1 --stage1-recovery-run $R1
  python -m evidence_pipeline.merge --stage $STAGE --dataset Beauty --run-name qwen3.5-9b__$REV \
      --stage1-run $S1 --stage1-recovery-run $R1
  python -m evidence_pipeline.run_recovery --stage $STAGE --dataset Beauty --model-config $M2 \
      --run-name qwen3.5-9b__$REV --stage1-run $S1 --stage1-recovery-run $R1
  python -m evidence_pipeline.merge_recovery --stage $STAGE --dataset Beauty --run-name qwen3.5-9b__$REV \
      --recovery-run-name qwen3.5-9b__failure-recovery-v1__thinking --stage1-run $S1 --stage1-recovery-run $R1
done

python -m evidence_pipeline.compose --dataset Beauty --stage1-run $S1 --stage1-recovery-run $R1 \
    --stage2-run qwen3.5-9b__projection-v1 --stage2-recovery-run qwen3.5-9b__failure-recovery-v1__thinking \
    --stage3-run qwen3.5-9b__stance-v1 --stage3-recovery-run qwen3.5-9b__failure-recovery-v1__thinking [--strict]
```

Then link the composed run as `final` (see the layout above) for the clustering stage.

## Diagnostics and resuming

```bash
python -m evidence_pipeline.summary --stage 1 --dataset Beauty --run-name $S1 [--recovery-run $R1 --resolved]
python -m evidence_pipeline.inspect_results --stage 1 --dataset Beauty --run-name $S1 [--status error] [--n 20]
PYTHONPATH=. python evidence_pipeline/scripts/make_eval_indices.py --dataset Beauty --n 200   # fixed review sample (--indices-file)
```

Rerunning an interrupted pass with the same command resumes it: every key already attempted (`ok`, `partial`,
`error`) is skipped (failures are left to recovery), and a truncated last JSONL line left by a timeout is moved to
a `*.corrupt-tail-<time>.bak` file. Keep the same number of shards when resuming; `--force` restarts the shard from
scratch. `manifest.json` counts are only final after the merge step (`wc -l <run>/shards/*.jsonl` shows progress).

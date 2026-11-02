from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from statement_ranking.audit import audit_interactions
from statement_ranking.dataset import (
    assert_inputs,
    optional_input_paths,
    required_input_paths,
    resolve_dataset_paths,
)
from statement_ranking.export import export_final
from statement_ranking.kcore import iterative_k_core
from statement_ranking.prepare import iter_prepared_interactions
from statement_ranking.split import user_last_temporal_split
from statement_ranking.storage import (
    STAGE_FINAL, STAGE_KCORE, STAGE_PREPARE, STAGE_SPLIT,
    kcore_interactions_path, prepare_interactions_path, split_excluded_path, split_interactions_path,
)
from statement_ranking.utils import (
    atomic_write_json, quick_file_fingerprint, read_jsonl, reset_dir, stable_hash, write_jsonl,
)

STAGES = ("prepare", "kcore", "split", "finalize")


@dataclass(frozen=True)
class RunPlan:
    start: int
    stop: int
    force_from: int | None

    def contains(self, stage: str) -> bool:
        idx = STAGES.index(stage)
        return self.start <= idx <= self.stop

    def force(self, stage: str) -> bool:
        return self.force_from is not None and self.force_from <= STAGES.index(stage) <= self.stop


def build_plan(*, stage: str | None, from_stage: str | None, to_stage: str | None, force: bool) -> RunPlan:
    if stage and from_stage:
        raise ValueError("Use either --stage or --from-stage")
    if stage:
        if stage == "all":
            return RunPlan(0, len(STAGES) - 1, 0 if force else None)
        idx = STAGES.index(stage)
        return RunPlan(0, idx, idx if force else None)
    start = STAGES.index(from_stage) if from_stage else 0
    stop = STAGES.index(to_stage) if to_stage else len(STAGES) - 1
    if stop < start:
        raise ValueError("--to-stage must not precede --from-stage")
    return RunPlan(0, stop, start if from_stage else (0 if force else None))


def _manifest_current(path: Path, artifact_id: str, outputs: list[Path]) -> bool:
    if not path.exists() or any(not p.exists() for p in outputs):
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(value.get("complete")) and value.get("artifact_id") == artifact_id


def _input_signature(paths) -> dict[str, Any]:
    return {
        "required": [quick_file_fingerprint(p) for p in required_input_paths(paths)],
        "optional_present": [quick_file_fingerprint(p) for p in optional_input_paths(paths) if p.exists()],
    }


def run_pipeline(
    *, dataset: str, cfg: dict[str, Any], plan: RunPlan,
    dataset_root: str | Path | None = None, output_root: str | Path | None = None,
) -> None:
    paths = resolve_dataset_paths(dataset, cfg, dataset_root=dataset_root, output_root=output_root)
    assert_inputs(paths)
    root = paths.statement_ranking_root
    root.mkdir(parents=True, exist_ok=True)

    if plan.contains("prepare"):
        stage_root = root / STAGE_PREPARE
        out = prepare_interactions_path(root)
        artifact = stable_hash({
            "stage": "prepare", "implementation": "statement-ranking-final-only-v2",
            "inputs": _input_signature(paths), "selection": cfg.get("selection", {}),
        })
        manifest_path = stage_root / "manifest.json"
        if plan.force("prepare") or not _manifest_current(manifest_path, artifact, [out]):
            reset_dir(stage_root)
            iterator, stats = iter_prepared_interactions(paths, cfg)
            write_jsonl(out, iterator)
            atomic_write_json(manifest_path, {
                "complete": True, "artifact_id": artifact, "stats": stats,
                "upstream_intermediate_artifacts_required": False,
            })
            print(f"prepare: retained={stats['retained_match_interactions']} / input={stats['input_interactions']}")
        else:
            print("prepare: up to date")

    if plan.contains("kcore"):
        stage_root = root / STAGE_KCORE
        src = prepare_interactions_path(root)
        out = kcore_interactions_path(root)
        artifact = stable_hash({
            "stage": "kcore", "implementation": "iterative-bipartite-kcore-v1",
            "source": quick_file_fingerprint(src), "k_core": cfg.get("k_core", {}),
        })
        manifest_path = stage_root / "manifest.json"
        if plan.force("kcore") or not _manifest_current(manifest_path, artifact, [out]):
            reset_dir(stage_root)
            rows = [record for _, record in read_jsonl(src)]
            filtered, stats = iterative_k_core(rows, int(cfg.get("k_core", {}).get("k", 5)))
            write_jsonl(out, filtered)
            atomic_write_json(manifest_path, {"complete": True, "artifact_id": artifact, "stats": stats})
            print(f"kcore: {len(rows)} -> {len(filtered)} interactions")
        else:
            print("kcore: up to date")

    if plan.contains("split"):
        stage_root = root / STAGE_SPLIT
        src = kcore_interactions_path(root)
        out = split_interactions_path(root)
        excluded_out = split_excluded_path(root)
        split_cfg = cfg.get("split", {})
        artifact = stable_hash({
            "stage": "split", "implementation": "user-last-temporal-v2",
            "source": quick_file_fingerprint(src), "split": split_cfg,
        })
        manifest_path = stage_root / "manifest.json"
        if plan.force("split") or not _manifest_current(manifest_path, artifact, [out, excluded_out]):
            reset_dir(stage_root)
            rows = [record for _, record in read_jsonl(src)]
            split_rows, excluded, stats = user_last_temporal_split(
                rows, tie_breaker=str(split_cfg.get("tie_breaker", "interaction_id")),
            )
            write_jsonl(out, split_rows)
            write_jsonl(excluded_out, excluded)
            atomic_write_json(manifest_path, {"complete": True, "artifact_id": artifact, "stats": stats})
            print(
                f"split: train={stats['train_interactions']} test={stats['test_interactions']} "
                f"cold-start-excluded={stats['excluded_cold_start_test_interactions']}"
            )
        else:
            print("split: up to date")

    if plan.contains("finalize"):
        src = split_interactions_path(root)
        excluded_src = split_excluded_path(root)
        final_root = root / STAGE_FINAL
        artifact = stable_hash({
            "stage": "finalize", "implementation": "statement-ranking-csv-v2-final-only",
            "source": quick_file_fingerprint(src), "excluded": quick_file_fingerprint(excluded_src),
            "optional_sources": [quick_file_fingerprint(p) for p in optional_input_paths(paths) if p.exists()],
        })
        marker = final_root / ".artifact.json"
        required = [final_root / "manifest.json", final_root / "interactions_match.csv"]
        if plan.force("finalize") or not _manifest_current(marker, artifact, required):
            reset_dir(final_root)
            rows = [record for _, record in read_jsonl(src)]
            excluded = [record for _, record in read_jsonl(excluded_src)]
            prepare_stats = json.loads((root / STAGE_PREPARE / "manifest.json").read_text(encoding="utf-8"))["stats"]
            kcore_stats = json.loads((root / STAGE_KCORE / "manifest.json").read_text(encoding="utf-8"))["stats"]
            split_stats = json.loads((root / STAGE_SPLIT / "manifest.json").read_text(encoding="utf-8"))["stats"]
            export_final(
                paths=paths, output_root=root, interactions=rows, excluded=excluded,
                pipeline_stats={"prepare": prepare_stats, "kcore": kcore_stats, "split": split_stats},
            )
            kcore_rows = [record for _, record in read_jsonl(kcore_interactions_path(root))]
            audit = audit_interactions(
                rows, kcore_rows, int(cfg.get("k_core", {}).get("k", 5)), paths=paths,
            )
            atomic_write_json(final_root / "audit.json", audit)
            if not audit["complete"]:
                raise RuntimeError("statement_ranking audit failed: " + "; ".join(audit["errors"]))
            atomic_write_json(marker, {"complete": True, "artifact_id": artifact})
            print(f"finalize: wrote {final_root}")
        else:
            print("finalize: up to date")

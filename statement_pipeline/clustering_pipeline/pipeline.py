from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clustering_pipeline.storage import TYPES

STAGES = ("prepare", "embed", "neighbors", "rerank", "cluster", "finalize", "audit")


@dataclass(frozen=True)
class RunPlan:
    start: int
    stop: int
    force_from: int | None

    def contains(self, stage: str) -> bool:
        idx = STAGES.index(stage)
        return self.start <= idx <= self.stop

    def force(self, stage: str) -> bool:
        if self.force_from is None:
            return False
        idx = STAGES.index(stage)
        return self.force_from <= idx <= self.stop


def build_plan(
    *,
    stage: str | None,
    from_stage: str | None,
    to_stage: str | None,
    force: bool,
) -> RunPlan:
    if stage and from_stage:
        raise ValueError("Use either --stage or --from-stage, not both")
    if stage and to_stage:
        raise ValueError("--to-stage cannot be combined with --stage; omit --stage for a resumable run through --to-stage")

    if stage:
        if stage == "all":
            start, stop = 0, len(STAGES) - 1
            force_from = 0 if force else None
        else:
            idx = STAGES.index(stage)
            # Dependencies are validated/reused automatically. Only the target
            # stage is forced when --force is supplied.
            start, stop = 0, idx
            force_from = idx if force else None
        return RunPlan(start=start, stop=stop, force_from=force_from)

    if from_stage:
        start_idx = STAGES.index(from_stage)
        stop_idx = STAGES.index(to_stage) if to_stage else len(STAGES) - 1
        if stop_idx < start_idx:
            raise ValueError("--to-stage must not precede --from-stage")
        # Stages before start_idx are still validated/reused as dependencies;
        # start_idx and every requested descendant are forced.
        return RunPlan(start=0, stop=stop_idx, force_from=start_idx)

    stop_idx = STAGES.index(to_stage) if to_stage else len(STAGES) - 1
    return RunPlan(start=0, stop=stop_idx, force_from=0 if force else None)


def run_pipeline(
    *,
    input_file: Path,
    output_root: Path,
    cfg: dict[str, Any],
    types: tuple[str, ...],
    plan: RunPlan,
    audit_ranks: str | None = None,
) -> None:
    def banner(name: str) -> None:
        mode = "FORCE" if plan.force(name) else "reuse/compute"
        print(f"\n=== {name} [{mode}] ===", flush=True)

    if plan.contains("prepare"):
        from clustering_pipeline.stages.prepare import run_prepare

        banner("prepare")
        run_prepare(
            input_file=input_file,
            output_root=output_root,
            cfg=cfg,
            force=plan.force("prepare"),
        )

    if plan.contains("embed"):
        from clustering_pipeline.stages.embed import run_embed

        banner("embed")
        for typ in types:
            run_embed(output_root=output_root, cfg=cfg, statement_type=typ, force=plan.force("embed"))

    if plan.contains("neighbors"):
        from clustering_pipeline.stages.neighbors import run_neighbors

        banner("neighbors")
        for typ in types:
            run_neighbors(output_root=output_root, cfg=cfg, statement_type=typ, force=plan.force("neighbors"))

    if plan.contains("rerank"):
        from clustering_pipeline.stages.rerank import run_rerank

        banner("rerank")
        for typ in types:
            run_rerank(output_root=output_root, cfg=cfg, statement_type=typ, force=plan.force("rerank"))

    if plan.contains("cluster"):
        from clustering_pipeline.stages.cluster import run_cluster

        banner("cluster")
        for typ in types:
            run_cluster(output_root=output_root, cfg=cfg, statement_type=typ, force=plan.force("cluster"))

    if plan.contains("finalize"):
        if set(types) != set(TYPES):
            raise ValueError("finalize requires all three types: user item match")
        from clustering_pipeline.stages.finalize import run_finalize

        banner("finalize")
        run_finalize(output_root=output_root, cfg=cfg, force=plan.force("finalize"))

    if plan.contains("audit") and bool((cfg.get("pair_analysis") or {}).get("enabled", True)):
        from clustering_pipeline.stages.audit import run_pair_analysis

        banner("audit")
        for typ in types:
            run_pair_analysis(
                output_root=output_root,
                cfg=cfg,
                statement_type=typ,
                force=plan.force("audit"),
                rank_mode_override=audit_ranks,
            )

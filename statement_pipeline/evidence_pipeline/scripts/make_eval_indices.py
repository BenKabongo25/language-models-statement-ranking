from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from evidence_pipeline.dataset import dataset_paths, load_pipeline_config
from evidence_pipeline.utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic review sample for staged extraction experiments.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nonempty-only", action="store_true")
    parser.add_argument("--pipeline-config", default="evidence_pipeline/configs/pipeline.yaml")
    args = parser.parse_args()
    cfg = load_pipeline_config(args.pipeline_config)
    candidates = []
    for index, review in read_jsonl(dataset_paths(args.dataset, cfg)["reviews"]):
        if args.nonempty_only and not str(review.get("review", "")).strip():
            continue
        candidates.append(index)
    if args.n < 1 or args.n > len(candidates):
        parser.error(f"n must be in [1, {len(candidates)}]")
    selected = sorted(random.Random(args.seed).sample(candidates, args.n))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} indices to {out}")


if __name__ == "__main__":
    main()

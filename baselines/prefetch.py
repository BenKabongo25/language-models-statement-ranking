from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_merged


def main() -> None:
    p = argparse.ArgumentParser(description="Prefetch Transformer weights required by one baseline config.")
    p.add_argument("--config", action="append", required=True, help="Merged in order; pass base then model/override configs.")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    cfg = load_merged(args.config, args.set)
    names: list[str] = []
    model = cfg.get("model", {})
    if model.get("transformer"):
        names.append(str(model["transformer"]))
    if not names:
        print("No Transformer checkpoint required by this baseline.")
        return
    from huggingface_hub import snapshot_download
    for name in dict.fromkeys(names):
        print(snapshot_download(repo_id=name))


if __name__ == "__main__":
    main()

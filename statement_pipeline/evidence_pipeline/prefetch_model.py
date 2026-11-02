from __future__ import annotations

import argparse
import os

from evidence_pipeline.model_runtime import assert_model_access, load_model_spec
from evidence_pipeline.utils import project_root, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefetch one configured Hugging Face model into the shared cache.")
    parser.add_argument("--model-config", required=True)
    args = parser.parse_args()

    spec = load_model_spec(resolve_path(project_root(), args.model_config))
    assert_model_access(spec)
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": spec.name,
        "token": os.environ.get(spec.hf_token_env) or None,
    }
    if spec.download_dir:
        kwargs["cache_dir"] = spec.download_dir
    try:
        path = snapshot_download(**kwargs)
    except Exception as exc:
        name = type(exc).__name__
        if name in {"GatedRepoError", "HfHubHTTPError"} or "401" in str(exc) or "403" in str(exc):
            raise RuntimeError(
                f"Cannot access {spec.name}. Ensure model access is granted and {spec.hf_token_env} contains a valid token."
            ) from exc
        raise
    print(path)


if __name__ == "__main__":
    main()

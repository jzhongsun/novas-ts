"""
ModelScope upload helper.

Upload *.parquet files from a local directory to a ModelScope dataset/model
repository via the modelscope SDK.

Requires:
    pip install modelscope
    export MODELSCOPE_ACCESS_TOKEN=...   # https://modelscope.cn/my/myaccesstoken
"""

import os
from pathlib import Path


def upload_to_modelscope(
    local_dir: str = "./output",
    repo_id: str = "",
    repo_type: str = "dataset",
    path_in_repo: str = "stock-day",
    allow_patterns: str | list[str] = "*.parquet",
    access_token: str | None = None,
) -> None:
    """Upload parquet files from local_dir to a ModelScope repository.

    Args:
        local_dir: Local directory containing parquet files to upload.
        repo_id: ModelScope repository, e.g. "your_username/stock-data".
        repo_type: "dataset" (default) or "model".
        path_in_repo: Target path within the repo.
        allow_patterns: Glob pattern(s) to select files (default "*.parquet").
        access_token: Token. If None, reads from MODELSCOPE_ACCESS_TOKEN env var.
    """
    try:
        from modelscope.hub.api import HubApi
    except ImportError:
        raise ImportError("pip install modelscope")

    if not repo_id:
        raise RuntimeError("repo_id is required (e.g. yourname/stock-data)")

    if access_token is None:
        access_token = os.environ.get("MODELSCOPE_ACCESS_TOKEN", "")
    if not access_token:
        raise RuntimeError("Set MODELSCOPE_ACCESS_TOKEN env var")

    api = HubApi()
    api.login(access_token)

    # Count matching files for the log (api.upload_folder does the actual filter)
    if isinstance(allow_patterns, str):
        pats = [allow_patterns]
    else:
        pats = list(allow_patterns)
    files = []
    for pat in pats:
        files.extend(Path(local_dir).glob(pat))
    n_files = len(set(files))
    if n_files == 0:
        print(f"No files matching {pats} in {local_dir}")
        return

    print(f"Uploading {n_files} files to modelscope:{repo_id} ({path_in_repo})")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=local_dir,
        path_in_repo=path_in_repo,
        allow_patterns=allow_patterns,
        commit_message=f"Update {path_in_repo or 'root'} data ({n_files} files)",
        repo_type=repo_type,
        token=access_token,
    )
    print(f"Done: https://www.modelscope.cn/{repo_type}s/{repo_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload parquet files to ModelScope")
    parser.add_argument("--local-dir", default="./output", help="Local directory to upload")
    parser.add_argument("--repo-id", required=True, help="ModelScope repo (e.g. yourname/stock-data)")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    parser.add_argument("--path-in-repo", default="stock-day")
    parser.add_argument("--allow-patterns", default="*.parquet", help="Glob pattern(s), comma-separated")
    args = parser.parse_args()

    upload_to_modelscope(
        local_dir=args.local_dir,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        path_in_repo=args.path_in_repo,
        allow_patterns=args.allow_patterns,
    )

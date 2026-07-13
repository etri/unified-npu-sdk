import argparse
import os
from pathlib import Path
import sys


def _is_repo_root(path: Path) -> bool:
    return (path / "src" / "unified_sdk").is_dir() and (path / "examples").is_dir()


def _resolve_repo_root() -> Path:
    env_root = os.getenv("UNIFIED_SDK_REPO_ROOT")
    if env_root:
        candidate = Path(env_root).resolve()
        if _is_repo_root(candidate):
            return candidate

    cwd = Path.cwd().resolve()
    if _is_repo_root(cwd):
        return cwd

    file_root = Path(__file__).resolve().parents[1]
    if _is_repo_root(file_root):
        return file_root

    for candidate in (Path("/workspace/unified-sdk"), Path("/workspace/unified-npu-sdk")):
        if _is_repo_root(candidate):
            return candidate

    return file_root


REPO_ROOT = _resolve_repo_root()
DEFAULT_MODEL = os.getenv("RNGD_MODEL", "furiosa-ai/Qwen2.5-0.5B-Instruct")
DEFAULT_MODELS_DIR = REPO_ROOT / "models"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a supported RNGD model snapshot into the repo-local models/ directory for custom FXB smoke."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Base directory where the local model snapshot will be stored.",
    )
    parser.add_argument(
        "--local-name",
        default=None,
        help="Destination subdirectory name. Defaults to the last path component of --model.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision (branch/tag/commit).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the destination directory already exists.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    model = str(args.model).strip()
    if not model:
        raise SystemExit("Error: --model must not be empty")

    local_name = args.local_name or Path(model).name or "model"
    models_dir = args.models_dir.expanduser().resolve()
    dest = models_dir / local_name

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub is required. Install it with `python3 -m pip install --user -U huggingface_hub`.")
        sys.exit(1)

    if dest.exists() and any(dest.iterdir()) and not args.force:
        print("Local model snapshot already exists.")
        print(f"(repo_root={REPO_ROOT})")
        print(f"(model={model})")
        print(f"(local_path={dest})")
        sys.exit(0)

    dest.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "repo_id": model,
        "local_dir": str(dest),
        "local_dir_use_symlinks": False,
    }
    if args.revision:
        kwargs["revision"] = args.revision
    if args.force:
        kwargs["force_download"] = True

    snapshot_download(**kwargs)

    print("Complete!", dest)
    print(f"(repo_root={REPO_ROOT})")
    print(f"(model={model})")
    print(f"(local_path={dest})")

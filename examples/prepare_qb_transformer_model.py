import argparse
from pathlib import Path
import shutil
import sys
import os


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
MODELS_DIR = REPO_ROOT / "models"
DEFAULT_MODEL_ID = os.getenv("QB_LLM_MODEL_ID", "mobilint/Llama-3.2-1B-Instruct")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download a precompiled transformer/LLM MXQ from the Mobilint Hugging Face group "
            "and normalize it into ./models for QB LLM smoke."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="예: mobilint/Llama-3.2-1B-Instruct")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="repo 내부 모델 자산 디렉터리")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="snapshot을 저장할 디렉터리 (기본: models/<repo-name>)",
    )
    parser.add_argument(
        "--output-mxq",
        type=Path,
        default=None,
        help="정규화된 .mxq 출력 경로 (기본: models/<repo-name>.mxq)",
    )
    parser.add_argument(
        "--pattern",
        default="*.mxq",
        help="snapshot 내부에서 찾을 MXQ glob 패턴 (기본: *.mxq)",
    )
    return parser


def _normalize_repo_name(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def _find_mxq(local_dir: Path, pattern: str) -> Path:
    matches = sorted(local_dir.rglob(pattern))
    matches = [p for p in matches if p.suffix == ".mxq"]
    if not matches:
        raise FileNotFoundError(
            f"No MXQ file matching {pattern!r} was found under {local_dir}. "
            "Inspect the downloaded snapshot and adjust --pattern if needed."
        )
    return matches[0]


if __name__ == "__main__":
    args = _build_parser().parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "Error: huggingface_hub is required. Rebuild the qb-only image or install the package first."
        )

    repo_name = _normalize_repo_name(args.model_id)
    models_dir = args.models_dir.expanduser().resolve()
    local_dir = (
        args.local_dir.expanduser().resolve()
        if args.local_dir is not None
        else (models_dir / repo_name).resolve()
    )
    output_mxq = (
        args.output_mxq.expanduser().resolve()
        if args.output_mxq is not None
        else (models_dir / f"{repo_name}.mxq").resolve()
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    output_mxq.parent.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.model_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=[args.pattern, "*.json", "*.md", "*.txt"],
    )

    mxq_path = _find_mxq(local_dir, args.pattern)
    if mxq_path.resolve() != output_mxq.resolve():
        shutil.copyfile(mxq_path, output_mxq)

    print("== QB transformer model ready ==")
    print(f"(repo_root={REPO_ROOT})")
    print(f"(model_id={args.model_id})")
    print(f"(local_snapshot={local_dir})")
    print(f"(resolved_mxq={mxq_path})")
    print(f"(normalized_mxq={output_mxq})")

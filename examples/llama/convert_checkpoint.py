from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


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

    file_root = Path(__file__).resolve().parents[2]
    if _is_repo_root(file_root):
        return file_root

    for candidate in (Path("/workspace/unified-sdk"), Path("/workspace/unified-npu-sdk")):
        if _is_repo_root(candidate):
            return candidate

    return file_root


REPO_ROOT = _resolve_repo_root()


def _vendor_root_candidates(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []
    env_root = os.getenv("TENSORRT_LLM_SRC")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.extend(
        [
            repo_root.parent / "TensorRT-LLM",
            repo_root / "TensorRT-LLM",
            Path("/workspace/TensorRT-LLM"),
            Path("/workspace/tensorrt_llm"),
            Path("/opt/TensorRT-LLM"),
        ]
    )

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _find_vendor_convert_script(repo_root: Path) -> Path:
    for root in _vendor_root_candidates(repo_root):
        script = root / "examples" / "llama" / "convert_checkpoint.py"
        if script.is_file():
            return script.resolve()
    raise FileNotFoundError(
        "Could not find TensorRT-LLM's examples/llama/convert_checkpoint.py. "
        "Set TENSORRT_LLM_SRC to a matching TensorRT-LLM source checkout, or place the checkout in a common location "
        f"such as {repo_root.parent / 'TensorRT-LLM'}."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified SDK wrapper for TensorRT-LLM's LLaMA/TinyLlama convert_checkpoint.py. "
            "It keeps the smoke flow under unified-sdk/examples while delegating to a matching vendor source checkout."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tp_size", type=int, default=None)
    parser.add_argument("--pp_size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--load_by_shard", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args, passthrough = parser.parse_known_args()
    vendor_script = _find_vendor_convert_script(REPO_ROOT)

    cmd = [
        sys.executable,
        str(vendor_script),
        "--model_dir",
        args.model_dir,
        "--output_dir",
        args.output_dir,
        "--dtype",
        args.dtype,
    ]
    if args.tp_size is not None:
        cmd.extend(["--tp_size", str(args.tp_size)])
    if args.pp_size is not None:
        cmd.extend(["--pp_size", str(args.pp_size)])
    if args.workers is not None:
        cmd.extend(["--workers", str(args.workers)])
    if args.load_by_shard:
        cmd.append("--load_by_shard")
    cmd.extend(passthrough)

    print("== TensorRT-LLM checkpoint prepare ==")
    print(f"repo_root = {REPO_ROOT}")
    print(f"vendor_script = {vendor_script}")
    print(f"model_dir = {args.model_dir}")
    print(f"output_dir = {args.output_dir}")

    completed = subprocess.run(cmd)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

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

    # Prefer the public repo checkout placed next to unified-sdk, because that is
    # the documented prepare flow for this branch.
    candidates.extend(
        [
            repo_root.parent / "TensorRT-LLM",
            repo_root / "TensorRT-LLM",
            Path("/workspace/TensorRT-LLM"),
        ]
    )

    env_root = os.getenv("TENSORRT_LLM_SRC")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.extend(
        [
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
    attempted: list[str] = []
    for root in _vendor_root_candidates(repo_root):
        candidates = (
            root / "examples" / "models" / "core" / "llama" / "convert_checkpoint.py",
            root / "examples" / "llama" / "convert_checkpoint.py",
        )
        for script in candidates:
            attempted.append(str(script))
            if script.is_file():
                return script.resolve()

    raise FileNotFoundError(
        "Could not find TensorRT-LLM's official llama convert_checkpoint.py in a public repo checkout. "
        "For this branch, checkpoint prepare expects an official TensorRT-LLM source checkout, typically "
        "placed at `../TensorRT-LLM` relative to `unified-sdk/`. "
        f"Attempted: {attempted}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified SDK wrapper for TensorRT-LLM's official llama convert_checkpoint.py. "
            "This branch treats checkpoint prepare as a separate public-repo-backed phase before custom compile."
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, passthrough = parser.parse_known_args(argv)
    vendor_script = _find_vendor_convert_script(REPO_ROOT)

    print("== TensorRT-LLM checkpoint prepare ==")
    print(f"repo_root = {REPO_ROOT}")
    print(f"model_dir = {args.model_dir}")
    print(f"output_dir = {args.output_dir}")
    print(f"conversion_api = vendor source script: {vendor_script}")

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

    completed = subprocess.run(cmd)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

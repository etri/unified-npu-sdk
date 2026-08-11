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
    file_root = Path(__file__).resolve().parents[1]
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
            Path("/opt/TensorRT-LLM"),
            repo_root.parent / "TensorRT-LLM",
            repo_root / "TensorRT-LLM",
            Path("/workspace/TensorRT-LLM"),
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


def _find_vendor_convert_script(repo_root: Path, family: str) -> Path:
    family = family.lower()
    family_candidates = {
        "qwen": (
            ("examples", "models", "core", "qwen", "convert_checkpoint.py"),
        ),
        "llama": (
            ("examples", "models", "core", "llama", "convert_checkpoint.py"),
            ("examples", "llama", "convert_checkpoint.py"),
        ),
    }
    if family not in family_candidates:
        raise ValueError(f"Unsupported model family for checkpoint prepare: {family}")

    attempted: list[str] = []
    for root in _vendor_root_candidates(repo_root):
        for rel_parts in family_candidates[family]:
            script = root.joinpath(*rel_parts)
            attempted.append(str(script))
            if script.is_file():
                return script.resolve()

    raise FileNotFoundError(
        f"Could not find TensorRT-LLM's official {family} convert_checkpoint.py in the configured source roots. "
        f"Attempted: {attempted}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a local TensorRT-LLM checkpoint directory from a local Hugging Face model path. "
            "This helper keeps the user-facing flow under unified-sdk/examples and delegates to the official "
            "TensorRT-LLM public repo workflow behind the scenes."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--model-ref", required=True, help="Local HF model directory")
    parser.add_argument("--output-dir", required=True, help="Output TensorRT-LLM checkpoint directory")
    parser.add_argument(
        "--model-family",
        default="qwen",
        choices=("qwen", "llama"),
        help="TensorRT-LLM vendor example family to use for checkpoint prepare (default: qwen)",
    )
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--pp-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--load-by-shard", action="store_true")
    parser.add_argument("--tensorrt-llm-src", default=None, help="Optional override for the TensorRT-LLM public repo checkout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    env = os.environ.copy()
    if args.tensorrt_llm_src:
        env["TENSORRT_LLM_SRC"] = str(Path(args.tensorrt_llm_src).expanduser().resolve())
    else:
        bundled_checkout = Path("/opt/TensorRT-LLM")
        sibling_checkout = REPO_ROOT.parent / "TensorRT-LLM"
        if bundled_checkout.is_dir():
            env["TENSORRT_LLM_SRC"] = str(bundled_checkout.resolve())
        elif sibling_checkout.is_dir():
            env["TENSORRT_LLM_SRC"] = str(sibling_checkout.resolve())

    vendor_script = _find_vendor_convert_script(REPO_ROOT, args.model_family)
    cmd = [
        sys.executable,
        str(vendor_script),
        "--model_dir",
        args.model_ref,
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

    print("== TensorRT-LLM checkpoint prepare helper ==")
    print(f"repo_root = {REPO_ROOT}")
    print(f"model_ref = {args.model_ref}")
    print(f"output_dir = {args.output_dir}")
    print(f"model_family = {args.model_family}")
    print(f"tensorrt_llm_src = {env.get('TENSORRT_LLM_SRC', '(expected: /opt/TensorRT-LLM or ../TensorRT-LLM)')}")
    print(f"vendor_script = {vendor_script}")
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())

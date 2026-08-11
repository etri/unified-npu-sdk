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
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tp-size", type=int, default=None)
    parser.add_argument("--pp-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--load-by-shard", action="store_true")
    parser.add_argument("--tensorrt-llm-src", default=None, help="Optional override for the TensorRT-LLM public repo checkout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "examples" / "llama" / "convert_checkpoint.py"),
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

    env = os.environ.copy()
    if args.tensorrt_llm_src:
        env["TENSORRT_LLM_SRC"] = str(Path(args.tensorrt_llm_src).expanduser().resolve())
    else:
        default_checkout = REPO_ROOT.parent / "TensorRT-LLM"
        if default_checkout.is_dir():
            env["TENSORRT_LLM_SRC"] = str(default_checkout.resolve())

    print("== TensorRT-LLM checkpoint prepare helper ==")
    print(f"repo_root = {REPO_ROOT}")
    print(f"model_ref = {args.model_ref}")
    print(f"output_dir = {args.output_dir}")
    print(f"tensorrt_llm_src = {env.get('TENSORRT_LLM_SRC', '(expected: ../TensorRT-LLM public repo checkout)')}")
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())

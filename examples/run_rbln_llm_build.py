import argparse
from pathlib import Path
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
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_sdk.build.api import build_unified_LLM
from unified_sdk.options import RBLNLLMBuildOptions
from unified_sdk.types import LLMBuildConfig


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build/fetch an RBLN LLM artifact via Unified SDK.")
    parser.add_argument("--model", required=True, help="HF model id or local model path.")
    parser.add_argument("--build-mode", choices=("fetch", "optimum_compile"), default="fetch")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts")
    parser.add_argument("--model-name", default="model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--create-runtimes", action="store_true")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    cfg = LLMBuildConfig(
        backend="rbln",
        model_or_path=args.model,
        out_dir=str(args.out_dir),
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        num_devices=args.num_devices,
        backend_options=RBLNLLMBuildOptions(
            build_mode=args.build_mode,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
            rbln_create_runtimes=args.create_runtimes,
        ),
    )

    result = build_unified_LLM(cfg)
    print("Complete!", result.compiled_model_path)
    print(f"(repo_root={REPO_ROOT})")
    print(f"(build_mode={args.build_mode})")

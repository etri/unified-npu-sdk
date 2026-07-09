# examples/run_rngd_build.py
import argparse
from pathlib import Path
import sys
import os

def _is_repo_root(path: Path) -> bool:
    return (path / "src" / "unified_sdk").is_dir() and (path / "examples").is_dir()


def _resolve_repo_root() -> Path:
    """
    기준:
      1) 환경 변수 UNIFIED_SDK_REPO_ROOT 가 있으면 우선 사용
      2) 현재 작업 디렉터리가 repo root 구조면 그걸 사용
      3) 현재 파일 위치(.../examples/run_rngd_build.py) 기준으로 checkout root 추론
      4) 마지막 fallback 으로 알려진 컨테이너 경로를 확인
    """
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

try:
    from unified_sdk.types import BuildConfig
    from unified_sdk.build.api import build_unified
except ImportError:
    print("Error: 'unified_sdk' package not found. Install it first or run from the repository checkout.")
    sys.exit(1)
EXAMPLES_DIR = REPO_ROOT / "examples"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

DEFAULT_MODEL = os.getenv("RNGD_MODEL", "furiosa-ai/Qwen2.5-0.5B-Instruct")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a FuriosaAI RNGD model. "
        "기본은 HF 아티팩트/모델 id 확보(fetch), --compile 지정 시 ArtifactBuilder AOT 컴파일(compile hook)."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF 모델 id 또는 로컬 모델/아티팩트 경로.")
    parser.add_argument("--out-dir", type=Path, default=ARTIFACTS_DIR, help="AOT 아티팩트 출력 상위 디렉터리.")
    parser.add_argument("--model-name", default=None, help="아티팩트 하위 디렉터리 이름 (기본: 모델 id 의 마지막 요소).")
    parser.add_argument("--compile", action="store_true", help="ArtifactBuilder 로 AOT 컴파일 (compile hook).")
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("RNGD_TP", "1")))
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    model_name = args.model_name or Path(str(args.model)).name or "model"
    out_dir = args.out_dir.expanduser().resolve()

    cfg = BuildConfig(
        backend="rngd",
        model_or_path=str(args.model),
        out_dir=str(out_dir),
        model_name=model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        max_model_len=args.max_model_len,
        extra={"compile": bool(args.compile)},
    )

    result = build_unified(cfg)
    mode = "AOT compile (ArtifactBuilder)" if args.compile else "fetch (provided model id / artifact)"
    print("Complete!", result.compiled_model_path)
    print(f"(repo_root={REPO_ROOT})")
    print(f"(mode={mode})")

# examples/run_tensorrt_build.py
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
      3) 현재 파일 위치(.../examples/run_tensorrt_build.py) 기준으로 checkout root 추론
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
MODELS_DIR = REPO_ROOT / "models"
BUILD_OUTPUT_DIR = REPO_ROOT / "build_output"


def _parse_shape(value: str) -> tuple[int, ...]:
    parts = value.replace("x", ",").split(",")
    try:
        shape = tuple(int(part.strip()) for part in parts if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape: {value!r}") from exc
    if not shape or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError(f"shape must contain positive integers: {value!r}")
    return shape


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile an ONNX model to a TensorRT .engine.")
    parser.add_argument("--onnx", type=Path, default=None, help="ONNX 경로. 미지정 시 --models-dir 에서 탐색.")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="ONNX 를 찾을 디렉터리.")
    parser.add_argument("--out-dir", type=Path, default=BUILD_OUTPUT_DIR, help=".engine 출력 디렉터리.")
    parser.add_argument("--model-name", default="yolov7", help="확장자 없는 출력 모델 이름.")
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp32")
    parser.add_argument("--input-name", default="images", help="ONNX 입력 텐서 이름.")
    parser.add_argument("--min-shape", type=_parse_shape, default=None)
    parser.add_argument("--opt-shape", type=_parse_shape, default=None)
    parser.add_argument("--max-shape", type=_parse_shape, default=None)
    parser.add_argument(
        "--input-shape",
        type=_parse_shape,
        default=(1, 3, 640, 640),
        help="min/opt/max 미지정 시 셋 다 이 값으로 고정(static shape).",
    )
    parser.add_argument("--workspace-mib", type=int, default=None, help="TensorRT workspace memory pool (MiB).")
    parser.add_argument("--require-onnx", action="store_true", help="ONNX 를 못 찾으면 실패 처리.")
    return parser


def _find_onnx(models_dir: Path, model_name: str) -> Path | None:
    candidates = sorted(models_dir.glob(f"{model_name}*.onnx")) + sorted(models_dir.glob("*.onnx"))
    return candidates[0] if candidates else None


if __name__ == "__main__":
    args = _build_parser().parse_args()

    models_dir = args.models_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = args.onnx.expanduser().resolve() if args.onnx else _find_onnx(models_dir, args.model_name)
    if onnx_path is None:
        msg = (
            f"{models_dir} 에서 {args.model_name}*.onnx 를 찾지 못했습니다.\n"
            f"예) {models_dir / (args.model_name + '.onnx')} 를 배치하거나 --onnx 로 직접 지정하세요."
        )
        if args.require_onnx:
            raise FileNotFoundError(msg)
        print("[WARN] " + msg)
        sys.exit(1)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    lo = args.min_shape or args.input_shape
    opt = args.opt_shape or args.input_shape
    hi = args.max_shape or args.input_shape

    extra: dict = {}
    if args.workspace_mib:
        extra["workspace_mib"] = args.workspace_mib

    cfg = BuildConfig(
        backend="tensorrt",
        model_or_path=str(onnx_path),
        out_dir=str(out_dir),
        model_name=args.model_name,
        precision=args.precision,
        input_name=args.input_name,
        min_input_shape=lo,
        opt_input_shape=opt,
        max_input_shape=hi,
        extra=extra or None,
    )

    result = build_unified(cfg)
    print("Complete!", result.compiled_model_path)
    print(f"(repo_root={REPO_ROOT})")
    print(f"(onnx={onnx_path}, precision={args.precision}, profile={lo}/{opt}/{hi})")

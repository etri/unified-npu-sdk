# examples/run_qb_build.py
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
      3) 현재 파일 위치(.../examples/run_qb_build.py) 기준으로 checkout root 추론
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
    from unified_sdk.frontends import (
        list_model_zoo_models,
        resolve_qb_build_request,
    )
    from unified_sdk.options import QBBuildOptions
    from unified_sdk.types import BuildConfig
    from unified_sdk.build.api import build_unified
except ImportError:
    print("Error: 'unified_sdk' package not found. Install it first or run from the repository checkout.")
    sys.exit(1)
EXAMPLES_DIR = REPO_ROOT / "examples"
MODELS_DIR = REPO_ROOT / "models"
BUILDS_DIR = REPO_ROOT / "builds"


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
    parser = argparse.ArgumentParser(
        description="Build a Mobilint ARISE (QB) .mxq model. "
        "기본은 표준 model zoo / 로컬 .mxq 확보(fetch), "
        "--from-onnx 또는 --from-pth 지정 시 Mobilint compiler Python API(qubee/qbcompiler)로 컴파일(compile hook)."
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="fetch 모드에서 먼저 탐색할 .mxq 디렉터리.")
    parser.add_argument("--out-dir", type=Path, default=BUILDS_DIR, help="결과 .mxq 출력 디렉터리.")
    parser.add_argument("--model-name", default="resnet50", help="확장자 없는 출력 모델 이름.")
    parser.add_argument("--mxq", type=Path, default=None, help="이미 컴파일된 .mxq 를 직접 사용(fetch/provided).")
    parser.add_argument("--from-onnx", type=Path, default=None, help="이 ONNX 를 Mobilint compiler Python API(qubee/qbcompiler)로 .mxq 컴파일(compile hook).")
    parser.add_argument("--from-pth", type=Path, default=None, help="이 .pth/.pt weights를 ONNX로 export한 뒤 .mxq 컴파일(현재 resnet50 예제 지원).")
    parser.add_argument("--export-onnx-path", type=Path, default=None, help="--from-pth 사용 시 생성할 중간 ONNX 경로 (기본: models/<model-name>.onnx).")
    parser.add_argument("--calib", type=Path, default=None, help="Mobilint compiler calibration 데이터셋 메타 파일(.txt/.json).")
    parser.add_argument(
        "--quantize-method",
        choices=("percentile", "maxpercentile", "max", "kl"),
        default="percentile",
    )
    parser.add_argument(
        "--use-random-calib",
        action="store_true",
        help="calibration 데이터 없이 random calib 로 컴파일(smoke).",
    )
    parser.add_argument(
        "--require-mxq",
        action="store_true",
        help="fetch 모드에서 .mxq 를 못 찾으면 실패 처리.",
    )
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--input-shape", type=_parse_shape, default=(1, 3, 224, 224))
    parser.add_argument("--core-mode", default=os.getenv("MBLT_CORE_MODE", "global8"))
    parser.add_argument("--product", default=os.getenv("MBLT_PRODUCT", "aries"))
    parser.add_argument(
        "--list-model-zoo",
        action="store_true",
        help="설치된 mblt_model_zoo.vision 에서 사용 가능한 표준 fetch 모델 목록을 출력하고 종료합니다.",
    )
    parser.add_argument(
        "--target-device",
        default=os.getenv("MBLT_TARGET_DEVICE", ""),
        help="Mobilint compiler target device. 예: aries-rb, regulus-ra, regulus-rb. 기본은 --product 에서 추론.",
    )
    return parser

if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.list_model_zoo:
        items = list_model_zoo_models()
        if not items:
            print("mblt_model_zoo.vision 에서 조회 가능한 모델을 찾지 못했습니다.")
            print("mblt-model-zoo 패키지 설치 상태를 확인하세요.")
            sys.exit(1)
        print("== Available model zoo fetch targets ==")
        print("(normalized_name -> mblt_model_zoo.vision symbol)")
        for normalized, symbol in items:
            print(f"{normalized} -> {symbol}")
        sys.exit(0)

    models_dir = args.models_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_options = QBBuildOptions(
        quantize_method=args.quantize_method,
        use_random_calib=True if args.use_random_calib else None,
        calib_data_path=str(args.calib.expanduser().resolve()) if args.calib else None,
        product=args.product,
        target_device=args.target_device,
    )

    try:
        resolved_request = resolve_qb_build_request(
            model_name=args.model_name,
            models_dir=models_dir,
            product=args.product,
            core_mode=args.core_mode,
            from_pth=args.from_pth,
            from_onnx=args.from_onnx,
            provided_mxq=args.mxq,
            export_onnx_path=args.export_onnx_path,
            input_name=args.input_name,
            input_shape=args.input_shape,
            require_mxq=args.require_mxq,
        )
    except FileNotFoundError as exc:
        if args.require_mxq:
            raise
        print("[WARN] " + str(exc))
        sys.exit(1)

    model_or_path = resolved_request.model_or_path
    source_desc = resolved_request.source_description

    cfg = BuildConfig(
        backend="qb",
        model_or_path=model_or_path,
        out_dir=str(out_dir),
        model_name=args.model_name,
        input_name=args.input_name,
        input_shape=args.input_shape,
        backend_options=build_options,
    )

    result = build_unified(cfg)
    print("Complete!", result.compiled_model_path)
    print(f"(repo_root={REPO_ROOT})")
    print(f"(source={source_desc})")

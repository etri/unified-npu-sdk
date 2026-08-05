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
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from unified_sdk.build import build_unified
    from unified_sdk.frontends import (
        WarboyFrontendBuildRequest,
        describe_frontend_api_mapping,
        list_model_zoo_targets,
        resolve_warboy_build_request,
    )
    from unified_sdk.options import WarboyBuildOptions
    from unified_sdk.types import BuildConfig
except ImportError:
    print("Error: 'unified_sdk' package not found. Install it first or run from the repository checkout.")
    sys.exit(1)


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
        description=(
            "Build a FuriosaAI Warboy .enf model. "
            "Build core는 provided .enf 배치 또는 quantized ONNX -> .enf 컴파일만 담당하고, "
            "fetch/model-zoo 해석은 frontend helper가 담당합니다."
        )
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="frontend fetch가 .enf 를 찾거나 저장할 디렉터리")
    parser.add_argument("--out-dir", type=Path, default=BUILDS_DIR, help="결과 .enf 출력 디렉터리")
    parser.add_argument("--model-name", default="resnet50", help="확장자 없는 출력 모델 이름")
    parser.add_argument("--enf", type=Path, default=None, help="이미 컴파일된 .enf 를 직접 사용")
    parser.add_argument("--from-onnx", type=Path, default=None, help="quantized ONNX 를 furiosa-compiler 로 .enf 컴파일")
    parser.add_argument("--target-npu", choices=("warboy", "warboy-2pe"), default="warboy-2pe")
    parser.add_argument("--require-enf", action="store_true", help="fetch 모드에서 .enf 를 못 찾으면 실패 처리")
    parser.add_argument(
        "--compiler-config",
        action="append",
        default=[],
        help="furiosa-compiler 에 그대로 전달할 추가 옵션. 예: --compiler-config=--without-quant-dequant",
    )
    parser.add_argument(
        "--list-model-zoo",
        action="store_true",
        help="설치된 furiosa.models.vision model zoo 에서 사용 가능한 모델 이름 후보를 출력하고 종료합니다.",
    )
    parser.add_argument(
        "--describe-frontend",
        action="store_true",
        help="frontend prepare/fetch -> build 경로 매핑을 출력하고 종료합니다.",
    )
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--input-shape", type=_parse_shape, default=(1, 3, 224, 224))
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.list_model_zoo:
        items = list_model_zoo_targets()
        if not items:
            print("furiosa.models.vision model zoo 목록을 찾지 못했습니다.")
            sys.exit(1)
        print("== Available Furiosa model zoo targets ==")
        for name in items:
            print(name)
        sys.exit(0)

    if args.describe_frontend:
        print(describe_frontend_api_mapping())
        sys.exit(0)

    models_dir = args.models_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    request = WarboyFrontendBuildRequest(
        model_name=args.model_name,
        models_dir=models_dir,
        target_npu=args.target_npu,
        provided_enf=args.enf.expanduser().resolve() if args.enf else None,
        from_onnx=args.from_onnx.expanduser().resolve() if args.from_onnx else None,
        require_enf=args.require_enf,
    )
    resolved = resolve_warboy_build_request(request=request)

    prepared_input = resolved.prepared_input
    if prepared_input.kind == "provided_artifact":
        source_desc = f"{resolved.kind}: {prepared_input.provided_artifact.source_path}"
        model_or_path = str(prepared_input.provided_artifact.source_path)
    else:
        source_desc = f"{resolved.kind}: {prepared_input.compile_source.source}"
        model_or_path = str(prepared_input.compile_source.source)

    cfg = BuildConfig(
        backend="warboy",
        model_or_path=model_or_path,
        out_dir=str(out_dir),
        model_name=args.model_name,
        input_name=args.input_name,
        input_shape=args.input_shape,
        backend_options=WarboyBuildOptions(
            target_npu=args.target_npu,
            target_ir="enf",
            compiler_config=tuple(args.compiler_config),
        ),
        prepared_input=prepared_input,
    )

    result = build_unified(cfg)
    print("Complete!", result.compiled_model_path)
    print(f"(repo_root={REPO_ROOT})")
    print(f"(source={source_desc})")

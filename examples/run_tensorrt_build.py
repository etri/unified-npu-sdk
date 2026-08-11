import argparse
import os
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
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from unified_sdk.build.api import build_unified
    from unified_sdk.frontends import list_torchvision_model_zoo_targets, resolve_tensorrt_vision_build_request
    from unified_sdk.frontends.types import TensorRTVisionFrontendBuildRequest
    from unified_sdk.options import TensorRTVisionBuildOptions
    from unified_sdk.types import BuildConfig
except ImportError:
    print("Error: 'unified_sdk' package not found. Install it first or run from the repository checkout.")
    sys.exit(1)


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
    parser = argparse.ArgumentParser(
        description=(
            "Build/fetch a TensorRT .engine. "
            "기본은 torchvision model zoo standard fetch, --engine 은 provided artifact, "
            "--onnx / --from-pth 는 frontend prepare 후 custom compile 입니다."
        )
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--out-dir", type=Path, default=BUILD_OUTPUT_DIR)
    parser.add_argument("--model-name", default="resnet50")
    parser.add_argument("--engine", type=Path, default=None)
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--from-pth", type=Path, default=None)
    parser.add_argument("--export-onnx-path", type=Path, default=None)
    parser.add_argument("--list-model-zoo", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp32")
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--min-shape", type=_parse_shape, default=None)
    parser.add_argument("--opt-shape", type=_parse_shape, default=None)
    parser.add_argument("--max-shape", type=_parse_shape, default=None)
    parser.add_argument("--input-shape", type=_parse_shape, default=(1, 3, 224, 224))
    parser.add_argument("--workspace-mib", type=int, default=None)
    parser.add_argument("--require-onnx", action="store_true")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.list_model_zoo:
        items = list_torchvision_model_zoo_targets()
        if not items:
            print("torchvision model zoo 목록을 찾지 못했습니다.")
            sys.exit(0)
        print("== Available torchvision model zoo targets ==")
        for item in items:
            print(item)
        sys.exit(0)

    min_shape = args.min_shape or args.input_shape
    opt_shape = args.opt_shape or args.input_shape
    max_shape = args.max_shape or args.input_shape

    request = TensorRTVisionFrontendBuildRequest(
        model_name=args.model_name,
        models_dir=args.models_dir,
        out_dir=args.out_dir,
        precision=args.precision,
        provided_engine=args.engine,
        onnx_path=args.onnx,
        weights_path=args.from_pth,
        export_onnx_path=args.export_onnx_path,
        model_zoo_model=args.model_name if not any((args.engine, args.onnx, args.from_pth)) else None,
        pretrained=not any((args.engine, args.onnx, args.from_pth)),
        require_onnx=args.require_onnx,
        input_name=args.input_name,
        min_input_shape=min_shape,
        opt_input_shape=opt_shape,
        max_input_shape=max_shape,
    )
    resolved = resolve_tensorrt_vision_build_request(request)

    cfg = BuildConfig(
        backend="tensorrt",
        model_or_path=resolved.model_or_path,
        out_dir=args.out_dir,
        model_name=args.model_name,
        backend_options=TensorRTVisionBuildOptions(
            precision=args.precision,
            workspace_mib=args.workspace_mib,
        ),
        prepared_input=resolved.prepared_input,
    )
    result = build_unified(cfg)

    print("== TensorRT build ==")
    print(f"repo_root = {REPO_ROOT}")
    print(f"source = {resolved.source_description}")
    print(f"artifact = {result.compiled_model_path}")
    if result.meta_data:
        print(f"meta = {result.meta_data}")

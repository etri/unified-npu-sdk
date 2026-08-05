# examples/run_rbln_build.py
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

try:
    from unified_sdk.build.api import build_unified
    from unified_sdk.frontends import (
        RBLNVisionFrontendBuildRequest,
        list_model_zoo_targets,
        resolve_rbln_vision_build_request,
    )
    from unified_sdk.options import RBLNVisionBuildOptions
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


def _parse_bucketing_shapes(value: str | None) -> list[tuple[int, ...]] | None:
    if not value:
        return None
    return [_parse_shape(item) for item in value.split(";") if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build/fetch an RBLN vision artifact. Supports model-zoo/source fetch, provided .rbln fetch, "
            "reference/tutorial compile, PTH/PT restore, and experimental/unverified ONNX restore."
        )
    )
    parser.add_argument(
        "--list-model-zoo",
        action="store_true",
        help="Print supported RBLN model-zoo/source-fetch targets and exit.",
    )
    parser.add_argument(
        "--compiled-model-ref",
        default=None,
        help=(
            "Hub model id or local directory containing a precompiled RBLN artifact directory "
            "(*.rbln + rbln_config.json). Advanced helper for a precompiled artifact repository/directory."
        ),
    )
    parser.add_argument("--model-zoo-model", default=None, help="Reference model-zoo target name, e.g. resnet50.")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained torchvision weights for the selected model-zoo target.")
    parser.add_argument("--weights", type=Path, default=None, help="Legacy alias for --from-pth.")
    parser.add_argument("--from-pth", type=Path, default=None, help="Compile from a local .pth/.pt checkpoint by restoring a torchvision model.")
    parser.add_argument(
        "--from-onnx",
        type=Path,
        default=None,
        help=(
            "Experimental/unverified: restore a torch model from ONNX and compile to .rbln. "
            "This path is vendor-dependent and may fail or crash for some graphs."
        ),
    )
    parser.add_argument("--rbln", type=Path, default=None, help="Fetch a precompiled .rbln from a local path into the build output directory.")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="Directory used to find local model/checkpoint files.")
    parser.add_argument("--out-dir", type=Path, default=BUILDS_DIR, help="Directory for compiled/fetched .rbln output.")
    parser.add_argument("--model-name", default="resnet50", help="Output model name without extension.")
    parser.add_argument("--require-weights", action="store_true", help="Fail if no local ResNet50 weights are found for checkpoint-based build.")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--input-shape", type=_parse_shape, default=(1, 3, 224, 224))
    parser.add_argument(
        "--bucketing-shapes",
        default=None,
        help="Semicolon-separated input shapes, e.g. '1,3,224,224;4,3,224,224'.",
    )
    parser.add_argument(
        "--model-trace-method",
        choices=("export", "export_strict", "jittrace"),
        default=None,
        help="Optional RBLN trace method passed to compile_from_torch.",
    )
    parser.add_argument("--npu", default=os.getenv("RBLN_NPU_NAME", "RBLN-CA22"))
    return parser


def _check_repo_layout(models_dir: Path, out_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    args = _build_parser().parse_args()

    selected_modes = sum(
        1
        for value in (
            bool((args.compiled_model_ref or "").strip()),
            bool(args.rbln),
            bool(args.from_onnx),
            bool(args.from_pth or args.weights),
            bool((args.model_zoo_model or "").strip()),
        )
        if value
    )
    if selected_modes > 1:
        raise SystemExit(
            "Choose only one build source at a time: "
            "--compiled-model-ref, --rbln, --from-onnx, --from-pth/--weights, or --model-zoo-model."
        )

    if args.list_model_zoo:
        print("== Supported RBLN model-zoo/source-fetch targets ==")
        for key, value in sorted(list_model_zoo_targets().items()):
            print(f"{key}: {value['symbol']} ({value['note']})")
        raise SystemExit(0)

    models_dir = args.models_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    _check_repo_layout(models_dir, out_dir)

    weights_path = (args.from_pth or args.weights)
    request = RBLNVisionFrontendBuildRequest(
        model_name=args.model_name,
        models_dir=models_dir,
        compiled_model_ref=(args.compiled_model_ref or "").strip() or None,
        provided_rbln=args.rbln.expanduser().resolve() if args.rbln else None,
        from_onnx=args.from_onnx.expanduser().resolve() if args.from_onnx else None,
        weights_path=weights_path.expanduser().resolve() if weights_path else None,
        model_zoo_model=(args.model_zoo_model or "").strip() or None,
        pretrained=args.pretrained,
        require_weights=args.require_weights,
    )
    resolved = resolve_rbln_vision_build_request(request)

    cfg = BuildConfig(
        backend="rbln",
        model_or_path=resolved.model_or_path,
        out_dir=str(out_dir),
        model_name=args.model_name,
        input_name=args.input_name,
        input_shape=args.input_shape,
        bucketing_shapes=_parse_bucketing_shapes(args.bucketing_shapes),
        backend_options=RBLNVisionBuildOptions(
            npu=args.npu,
            precision=args.precision,
            model_trace_method=args.model_trace_method,
        ),
        prepared_input=resolved.prepared_input,
    )

    result = build_unified(cfg)
    print("Complete!", result.compiled_model_path)
    print(f"(repo_root={REPO_ROOT})")
    print(f"(source={resolved.source_description})")

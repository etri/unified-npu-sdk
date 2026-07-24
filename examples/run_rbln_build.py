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
    from unified_sdk.types import BuildConfig
    from unified_sdk.build.api import build_unified
except ImportError:
    print("Error: 'unified_sdk' package not found. Install it first or run from the repository checkout.")
    sys.exit(1)

MODELS_DIR = REPO_ROOT / "models"
BUILDS_DIR = REPO_ROOT / "builds"

_MODEL_ZOO_TARGETS = {
    "resnet50": {
        "symbol": "torchvision.models.resnet50",
        "note": "official RBLN PyTorch ResNet50 tutorial/reference compile baseline",
    },
}

_LOCAL_COMPILED_DIR_PREFIXES = {"artifacts", "builds", "models", ".", ".."}


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
            "Build/fetch an RBLN vision artifact. Supports compiled artifact fetch by model ref/local dir, "
            "provided .rbln fetch, reference/tutorial compile, PTH/PT restore, and experimental ONNX restore."
        )
    )
    parser.add_argument(
        "--list-model-zoo",
        action="store_true",
        help="Print supported RBLN model-zoo/reference compile targets and exit.",
    )
    parser.add_argument(
        "--compiled-model-ref",
        default=None,
        help=(
            "Hub model id or local directory containing a precompiled RBLN artifact directory "
            "(*.rbln + rbln_config.json). This is the standard fetch path when a compiled artifact "
            "repository/directory is available."
        ),
    )
    parser.add_argument("--model-zoo-model", default=None, help="Reference model-zoo target name, e.g. resnet50.")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained torchvision weights for the selected model-zoo target.")
    parser.add_argument("--weights", type=Path, default=None, help="Legacy alias for --from-pth.")
    parser.add_argument("--from-pth", type=Path, default=None, help="Compile from a local .pth/.pt checkpoint by restoring a torchvision model.")
    parser.add_argument("--from-onnx", type=Path, default=None, help="Experimental: restore a torch model from ONNX and compile to .rbln.")
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


def _find_weights(models_dir: Path) -> Path | None:
    candidates = sorted(models_dir.glob("resnet50*.pth")) + sorted(models_dir.glob("resnet50*.pt"))
    if not candidates:
        return None
    return candidates[0]


def _load_state_dict(path: Path, torch_module) -> dict:
    obj = torch_module.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            obj = obj["model_state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"가중치 파일 형식을 해석할 수 없습니다: {path}")

    cleaned = {}
    for k, v in obj.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


def _build_torchvision_resnet50(*, pretrained: bool):
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)
    model.eval()
    return model


def _looks_like_local_compiled_ref(value: str) -> bool:
    if not value:
        return False
    path = Path(value)
    first_part = path.parts[0] if path.parts else ""
    return path.is_absolute() or first_part in _LOCAL_COMPILED_DIR_PREFIXES


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
        print("== Supported RBLN reference compile targets ==")
        for key, value in sorted(_MODEL_ZOO_TARGETS.items()):
            print(f"{key}: {value['symbol']} ({value['note']})")
        raise SystemExit(0)

    models_dir = args.models_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    _check_repo_layout(models_dir, out_dir)

    weights_path = args.from_pth or args.weights
    if weights_path is not None:
        weights_path = weights_path.expanduser().resolve()

    build_input = None
    source_note = ""

    compiled_model_ref = (args.compiled_model_ref or "").strip()
    if compiled_model_ref:
        ref_path = Path(compiled_model_ref).expanduser()
        if ref_path.exists():
            build_input = str(ref_path.resolve())
            source_note = f"standard fetch from local compiled RBLN directory: {build_input}"
        elif _looks_like_local_compiled_ref(compiled_model_ref):
            raise FileNotFoundError(
                "compiled-model-ref was interpreted as a local compiled RBLN directory, "
                f"but it does not exist: {ref_path.resolve()}\n"
                "If you intended a Hugging Face repo id, pass an explicit repo id like "
                "'org/repo-name'. If you intended a local directory, ensure it contains "
                "*.rbln and rbln_config.json."
            )
        else:
            try:
                from huggingface_hub import snapshot_download
            except Exception as exc:
                raise RuntimeError(
                    "compiled-model-ref hub fetch requires `huggingface_hub`. Install it first."
                ) from exc

            local_dir = models_dir / compiled_model_ref.split("/")[-1]
            snapshot_path = snapshot_download(
                repo_id=compiled_model_ref,
                local_dir=str(local_dir),
            )
            build_input = str(Path(snapshot_path).resolve())
            source_note = f"standard fetch from compiled RBLN artifact repo: {compiled_model_ref} -> {build_input}"
    elif args.rbln:
        build_input = str(args.rbln.expanduser().resolve())
        source_note = f"provided .rbln fetch: {build_input}"
    elif args.from_onnx:
        build_input = str(args.from_onnx.expanduser().resolve())
        source_note = f"experimental ONNX restore -> .rbln: {build_input}"
    else:
        try:
            import torch
        except ImportError:
            print("Error: 'torch' is required for RBLN build example.")
            sys.exit(1)

        model_zoo_target = (args.model_zoo_model or "").strip().lower()
        if model_zoo_target:
            if model_zoo_target not in _MODEL_ZOO_TARGETS:
                raise SystemExit(
                    f"Unsupported model-zoo target: {args.model_zoo_model!r}. "
                    f"Try one of: {', '.join(sorted(_MODEL_ZOO_TARGETS))}"
                )
            if model_zoo_target == "resnet50":
                build_input = _build_torchvision_resnet50(pretrained=args.pretrained)
                source_note = (
                    "reference compile from official RBLN model-zoo/tutorial baseline: torchvision ResNet50 pretrained"
                    if args.pretrained
                    else "reference compile from official RBLN model-zoo/tutorial baseline: torchvision ResNet50 local/random-init"
                )
        else:
            # 3-b) user PTH/PT -> torch restore -> .rbln
            if weights_path is None:
                weights_path = _find_weights(models_dir)
            if args.require_weights and weights_path is None:
                raise FileNotFoundError(
                    f"{models_dir} 에서 resnet50 가중치 파일을 찾지 못했습니다.\n"
                    f"예) {models_dir/'resnet50.pth'} 또는 {models_dir/'resnet50_state_dict.pth'}"
                )

            build_input = _build_torchvision_resnet50(pretrained=False)
            if weights_path is not None:
                sd = _load_state_dict(weights_path, torch)
                build_input.load_state_dict(sd, strict=False)
                source_note = f"user PTH/PT -> torch restore -> .rbln: {weights_path}"
            else:
                source_note = "local torchvision ResNet50 random-init -> .rbln"

    cfg = BuildConfig(
        backend="rbln",
        model_or_path=build_input,
        out_dir=str(out_dir),
        model_name=args.model_name,
        precision=args.precision,
        input_name=args.input_name,
        input_shape=args.input_shape,
        bucketing_shapes=_parse_bucketing_shapes(args.bucketing_shapes),
        extra={
            key: value
            for key, value in {
                "npu": args.npu,
                "model_trace_method": args.model_trace_method,
            }.items()
            if value
        },
    )

    result = build_unified(cfg)
    print("Complete!", result.compiled_model_path)
    print(f"(repo_root={REPO_ROOT})")
    print(f"(source={source_note})")

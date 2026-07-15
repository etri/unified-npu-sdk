# examples/run_tensorrt_build.py
import argparse
from pathlib import Path
import sys
import os
import re
from typing import Any


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
        description="Build/fetch a TensorRT .engine. "
        "기본은 torchvision model zoo standard fetch, "
        "--engine 은 custom fetch, "
        "--onnx / --from-pth 는 custom compile 입니다."
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="모델 자산 디렉터리.")
    parser.add_argument("--out-dir", type=Path, default=BUILD_OUTPUT_DIR, help=".engine 출력 디렉터리.")
    parser.add_argument("--model-name", default="resnet50", help="확장자 없는 출력 모델 이름.")
    parser.add_argument("--engine", type=Path, default=None, help="이미 컴파일된 .engine 를 직접 사용(fetch/provided).")
    parser.add_argument("--onnx", type=Path, default=None, help="이 ONNX 를 TensorRT .engine 로 컴파일.")
    parser.add_argument("--from-pth", type=Path, default=None, help="이 .pth/.pt weights를 ONNX로 export한 뒤 .engine 컴파일.")
    parser.add_argument(
        "--export-onnx-path",
        type=Path,
        default=None,
        help="--from-pth 또는 표준 fetch 사용 시 생성할 중간 ONNX 경로 (기본: models/<model-name>.onnx).",
    )
    parser.add_argument(
        "--list-model-zoo",
        action="store_true",
        help="설치된 torchvision model zoo 에서 사용 가능한 모델 이름 후보를 출력하고 종료합니다.",
    )
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp32")
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--min-shape", type=_parse_shape, default=None)
    parser.add_argument("--opt-shape", type=_parse_shape, default=None)
    parser.add_argument("--max-shape", type=_parse_shape, default=None)
    parser.add_argument(
        "--input-shape",
        type=_parse_shape,
        default=(1, 3, 224, 224),
        help="min/opt/max 미지정 시 셋 다 이 값으로 고정(static shape).",
    )
    parser.add_argument("--workspace-mib", type=int, default=None, help="TensorRT workspace memory pool (MiB).")
    parser.add_argument("--require-onnx", action="store_true", help="ONNX 를 못 찾으면 실패 처리.")
    return parser


def _normalize_torchvision_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _list_torchvision_models() -> list[str]:
    try:
        from torchvision import models as tv_models
    except Exception:
        return []

    if hasattr(tv_models, "list_models"):
        try:
            return sorted(str(name) for name in tv_models.list_models())
        except Exception:
            pass

    names: list[str] = []
    for name in dir(tv_models):
        if name.startswith("_"):
            continue
        candidate = getattr(tv_models, name, None)
        if callable(candidate) and name.lower() == name:
            names.append(name)
    return sorted(set(names))


def _resolve_torchvision_model_name(model_name: str) -> str | None:
    normalized = _normalize_torchvision_name(model_name)
    for candidate in _list_torchvision_models():
        if _normalize_torchvision_name(candidate) == normalized:
            return candidate
    return None


def _find_onnx(models_dir: Path, model_name: str) -> Path | None:
    candidates = sorted(models_dir.glob(f"{model_name}*.onnx")) + sorted(models_dir.glob("*.onnx"))
    return candidates[0] if candidates else None


def _find_engine(models_dir: Path, model_name: str, precision: str) -> Path | None:
    candidates = sorted(models_dir.glob(f"{model_name}*{precision.upper()}*.engine"))
    candidates += sorted(models_dir.glob(f"{model_name}*.engine"))
    candidates += sorted(models_dir.glob("*.engine"))
    return candidates[0] if candidates else None


def _unwrap_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "weights", "model_state_dict"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                payload = nested
                break
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return payload


def _strip_known_prefixes(key: str) -> str:
    while True:
        updated = key
        for prefix in ("module.", "model.", "net.", "network."):
            if updated.startswith(prefix):
                updated = updated[len(prefix):]
        if updated == key:
            return key
        key = updated


def _score_prefix_strip(state_dict: dict[str, Any], prefix: str, expected_keys: set[str]) -> tuple[int, dict[str, Any]]:
    plen = len(prefix)
    transformed = {}
    hits = 0
    for key, value in state_dict.items():
        stripped = key[plen:] if key.startswith(prefix) else key
        transformed[stripped] = value
        if stripped in expected_keys:
            hits += 1
    return hits, transformed


def _align_state_dict_namespaces(state_dict: dict[str, Any], expected_keys: set[str]) -> dict[str, Any]:
    cleaned = {_strip_known_prefixes(k): v for k, v in state_dict.items()}
    base_hits = sum(1 for k in cleaned if k in expected_keys)
    best_hits = base_hits
    best = cleaned

    first_segments = sorted({k.split(".", 1)[0] for k in cleaned if "." in k})
    for seg in first_segments:
        prefix = seg + "."
        hits, transformed = _score_prefix_strip(cleaned, prefix, expected_keys)
        if hits > best_hits:
            best_hits = hits
            best = transformed

    return best


def _resolve_torchvision_model(model_name: str, *, pretrained: bool):
    try:
        from torchvision import models as tv_models
    except ImportError as exc:
        raise RuntimeError("torchvision is required for torchvision model zoo fetching and .pth export.") from exc

    resolved_name = _resolve_torchvision_model_name(model_name)
    if resolved_name is None:
        raise ValueError(
            f"Unsupported torchvision model name: {model_name!r}. "
            "Use --list-model-zoo to inspect available standard fetch targets."
        )

    if hasattr(tv_models, "get_model"):
        kwargs = {}
        if pretrained:
            if hasattr(tv_models, "get_model_weights"):
                weights_enum = tv_models.get_model_weights(resolved_name)
                kwargs["weights"] = weights_enum.DEFAULT
            else:
                kwargs["pretrained"] = True
        else:
            if hasattr(tv_models, "get_model_weights"):
                kwargs["weights"] = None
            else:
                kwargs["pretrained"] = False
        return resolved_name, tv_models.get_model(resolved_name, **kwargs)

    factory = getattr(tv_models, resolved_name, None)
    if not callable(factory):
        raise ValueError(f"Resolved torchvision model is not callable: {resolved_name}")
    return resolved_name, factory(pretrained=pretrained)


def _prepare_module_from_pth(weights_path: Path, model_name: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to compile from .pth/.pt weights.") from exc

    resolved_name, model = _resolve_torchvision_model(model_name, pretrained=False)
    checkpoint = torch.load(str(weights_path), map_location="cpu")
    state_dict = _unwrap_state_dict(checkpoint)
    aligned = _align_state_dict_namespaces(state_dict, set(model.state_dict().keys()))
    missing, unexpected = model.load_state_dict(aligned, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Failed to load {resolved_name} weights cleanly from checkpoint. "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    model.eval()
    return resolved_name, model


def _export_module_to_onnx(model, export_onnx_path: Path, input_name: str, input_shape: tuple[int, ...]) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to export ONNX.") from exc

    export_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(input_shape, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(export_onnx_path),
        input_names=[input_name],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
    )
    if not export_onnx_path.is_file():
        raise RuntimeError(f"ONNX export did not produce a file: {export_onnx_path}")
    return export_onnx_path


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.list_model_zoo:
        items = _list_torchvision_models()
        if not items:
            print("torchvision model zoo 목록을 찾지 못했습니다.")
            sys.exit(1)
        print("== Available torchvision model zoo targets ==")
        for name in items:
            print(name)
        sys.exit(0)

    models_dir = args.models_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    export_onnx_path = (
        args.export_onnx_path.expanduser().resolve()
        if args.export_onnx_path is not None
        else (models_dir / f"{args.model_name}.onnx").resolve()
    )

    lo = args.min_shape or args.input_shape
    opt = args.opt_shape or args.input_shape
    hi = args.max_shape or args.input_shape

    extra: dict = {}
    if args.workspace_mib:
        extra["workspace_mib"] = args.workspace_mib

    source_desc = ""
    model_or_path: str

    if args.from_pth is not None:
        weights_path = args.from_pth.expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"PTH/PT weights not found: {weights_path}")
        resolved_name, model = _prepare_module_from_pth(weights_path, args.model_name)
        onnx_path = _export_module_to_onnx(
            model=model,
            export_onnx_path=export_onnx_path,
            input_name=args.input_name,
            input_shape=args.input_shape,
        )
        model_or_path = str(onnx_path)
        source_desc = f"local weights -> ONNX export -> TensorRT compile: {weights_path} -> {onnx_path} ({resolved_name})"
    elif args.onnx is not None:
        onnx_path = args.onnx.expanduser().resolve()
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")
        model_or_path = str(onnx_path)
        source_desc = f"local/custom ONNX -> TensorRT compile: {onnx_path}"
    elif args.engine is not None:
        engine_path = args.engine.expanduser().resolve()
        if not engine_path.is_file():
            raise FileNotFoundError(f"Engine not found: {engine_path}")
        model_or_path = str(engine_path)
        source_desc = f"custom/local fetch from provided .engine: {engine_path}"
    else:
        onnx_path = _find_onnx(models_dir, args.model_name)
        if onnx_path is not None:
            model_or_path = str(onnx_path)
            source_desc = f"local/custom ONNX -> TensorRT compile: {onnx_path}"
        else:
            resolved_name, model = _resolve_torchvision_model(args.model_name, pretrained=True)
            onnx_path = _export_module_to_onnx(
                model=model,
                export_onnx_path=export_onnx_path,
                input_name=args.input_name,
                input_shape=args.input_shape,
            )
            model_or_path = str(onnx_path)
            source_desc = (
                f"standard fetch from torchvision model zoo -> ONNX export -> TensorRT compile: "
                f"{resolved_name} -> {onnx_path}"
            )

    cfg = BuildConfig(
        backend="tensorrt",
        model_or_path=str(model_or_path),
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
    print(f"(source={source_desc})")
    print(f"(precision={args.precision}, profile={lo}/{opt}/{hi})")

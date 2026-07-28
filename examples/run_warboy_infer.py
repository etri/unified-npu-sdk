import argparse
import timeit
from pathlib import Path
import sys
import os
import re


def _center_crop_resize_pil(image, size: int):
    width, height = image.size
    scale = 256 / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)))
    left = max((resized.width - size) // 2, 0)
    top = max((resized.height - size) // 2, 0)
    return resized.crop((left, top, left + size, top + size))


def _load_image_batch(image_path: Path, input_shape: tuple[int, ...], np_module):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    cropped = _center_crop_resize_pil(image, input_shape[-1])
    array = np_module.asarray(cropped, dtype=np_module.float32) / 255.0
    array = array.transpose(2, 0, 1)
    mean = np_module.asarray(IMAGENET_MEAN, dtype=np_module.float32)[:, None, None]
    std = np_module.asarray(IMAGENET_STD, dtype=np_module.float32)[:, None, None]
    normalized = (array - mean) / std
    return normalized[None, ...].astype(np_module.float32)


def _load_image_batch_uint8(image_path: Path, input_shape: tuple[int, ...], np_module):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    cropped = _center_crop_resize_pil(image, input_shape[-1])
    array = np_module.asarray(cropped, dtype=np_module.uint8)
    array = array.transpose(2, 0, 1)
    return array[None, ...].astype(np_module.uint8)

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


# ====== 경로 설정 (checkout root 기준, 컨테이너에서는 현재 마운트된 repo root) ======
ENGINE_PATH = REPO_ROOT / "builds" / "resnet50.enf"   # <- builds 기준
IMG_PATH = REPO_ROOT / "models" / "input.jpg"
LABELS_PATH = REPO_ROOT / "tests" / "imagenet_classes.txt"  # 있으면 사용, 없으면 cls_id만 출력


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
    parser = argparse.ArgumentParser(description="Run inference with a compiled FuriosaAI Warboy .enf model.")
    parser.add_argument("--engine-path", type=Path, default=ENGINE_PATH)
    parser.add_argument("--image", type=Path, default=IMG_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--output-name", default="output")
    parser.add_argument("--input-shape", type=_parse_shape, default=(1, 3, 224, 224))
    parser.add_argument("--device", default=os.getenv("FURIOSA_DEVICES", None),
                        help="예: 'warboy(0)*2'. 미지정 시 furiosa-runtime 기본 선택.")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--allow-dynamic-shape", action="store_true")
    return parser


def _check_files(engine_path: Path):
    missing = []
    if not engine_path.is_file():
        missing.append(f"- engine: {engine_path}")
    if missing:
        raise FileNotFoundError("필요한 파일이 없습니다:\n" + "\n".join(missing))


# ====== ImageNet 표준 preprocess (weights 없이 고정) ======
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _load_labels(labels_path: Path):
    if labels_path.is_file():
        labels = [l.strip() for l in labels_path.read_text().splitlines() if l.strip()]
        return labels
    return None


def _normalize_model_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _list_model_zoo_targets() -> list[str]:
    try:
        from furiosa.models import vision
    except Exception:
        return []

    declared = getattr(vision, "__all__", None)
    if isinstance(declared, (list, tuple)):
        return sorted({str(name) for name in declared if isinstance(name, str) and not name.startswith("_")})

    return sorted({name for name in dir(vision) if not name.startswith("_")})


def _resolve_model_zoo_target(model_name: str) -> str | None:
    normalized = _normalize_model_name(model_name)
    for candidate in _list_model_zoo_targets():
        if _normalize_model_name(candidate) == normalized:
            return candidate
    return None


def _prefers_uint8_fallback(model_name: str) -> bool:
    normalized = _normalize_model_name(model_name)
    # Detection / pose 계열 model zoo helper는 이미지가 없을 때도 uint8 입력 계약인 경우가 많다.
    return normalized.startswith(("yolov5", "yolov7", "ssd"))


def _maybe_create_model_zoo_helper(engine_path: Path):
    try:
        from furiosa.models import vision
    except Exception:
        return None

    resolved = _resolve_model_zoo_target(engine_path.stem)
    if resolved is None or not hasattr(vision, resolved):
        return None
    try:
        return getattr(vision, resolved)()
    except Exception:
        return None


def _load_with_model_zoo_preprocess(model_helper, image_path: Path, prefer_uint8: bool = False):
    preprocess_error = None
    kwargs_candidates = ({}, {"with_scaling": True}) if prefer_uint8 else ({"with_scaling": True}, {})
    for kwargs in kwargs_candidates:
        for candidate in ([str(image_path)], str(image_path)):
            try:
                inputs, contexts = model_helper.preprocess(candidate, **kwargs)
                arr = inputs[0] if isinstance(inputs, (list, tuple)) and len(inputs) == 1 else inputs
                dtype = getattr(arr, "dtype", None)
                if prefer_uint8 and dtype is not None and dtype != "uint8" and str(dtype) != "uint8":
                    preprocess_error = RuntimeError(
                        f"Model Zoo preprocess produced dtype={dtype!r} with kwargs={kwargs}, expected uint8-compatible input"
                    )
                    continue
                return inputs, contexts, kwargs
            except Exception as exc:
                preprocess_error = exc
    raise RuntimeError(f"Model Zoo preprocess failed: {preprocess_error!r}")


def _extract_prediction_id(output_array, torch_module) -> int:
    y_t = torch_module.from_numpy(output_array)
    if y_t.ndim == 1 and y_t.numel() == 1:
        return int(y_t.item())
    flat = y_t.reshape(y_t.shape[0], -1)
    return int(torch_module.argmax(flat, dim=1).item())


def _format_output_shape(output) -> str:
    if isinstance(output, list):
        return "[" + ", ".join(str(tuple(arr.shape)) for arr in output) + "]"
    return str(tuple(output.shape))


def _maybe_retry_uint8_batch(image_path: Path, input_shape: tuple[int, ...], model_helper, np_module):
    if not image_path.is_file():
        try:
            return (
                np_module.zeros(input_shape, dtype=np_module.uint8),
                None,
                {"synthetic_uint8_fallback": True},
            )
        except Exception:
            return None, None, None
    if model_helper is not None:
        try:
            batch, contexts, preprocess_kwargs = _load_with_model_zoo_preprocess(
                model_helper,
                image_path,
                prefer_uint8=True,
            )
            return batch, contexts, preprocess_kwargs
        except Exception:
            pass
    try:
        return (
            _load_image_batch_uint8(image_path, input_shape, np_module),
            None,
            {"manual_uint8_fallback": True},
        )
    except Exception:
        return None, None, None


if __name__ == "__main__":
    args = _build_parser().parse_args()

    try:
        import numpy as np
        import torch
        from PIL import Image  # noqa: F401
    except ImportError:
        print("Error: 'numpy', 'torch', and 'pillow' are required for the Warboy inference example.")
        sys.exit(1)

    from unified_sdk.types import RuntimeConfig
    from unified_sdk.runtime import create_runtime, infer, destroy_runtime

    engine_path = args.engine_path.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()

    if args.iters <= 0:
        raise ValueError("--iters must be > 0")

    _check_files(engine_path)
    labels = _load_labels(labels_path)
    model_helper = _maybe_create_model_zoo_helper(engine_path)
    contexts = None

    if image_path.is_file():
        if model_helper is not None:
            batch, contexts, preprocess_kwargs = _load_with_model_zoo_preprocess(
                model_helper,
                image_path,
                prefer_uint8=_prefers_uint8_fallback(engine_path.stem),
            )
            input_source = f"{image_path} (model-zoo preprocess {preprocess_kwargs})"
        else:
            batch = _load_image_batch(image_path, args.input_shape, np)
            input_source = str(image_path)
    else:
        if model_helper is not None:
            if _prefers_uint8_fallback(engine_path.stem):
                batch = np.zeros(args.input_shape, dtype=np.uint8)
                input_source = f"synthetic zeros uint8 {args.input_shape} (model-zoo fallback)"
            else:
                batch = torch.zeros(args.input_shape, dtype=torch.float32).numpy()
                input_source = f"synthetic zeros float32 {args.input_shape} (model-zoo fallback)"
        else:
            batch = torch.zeros(args.input_shape, dtype=torch.float32).numpy()
            input_source = f"synthetic zeros float32 {args.input_shape}"

    # NOTE: quantized ENF 의 입력 dtype/layout 은 컴파일 시 고정된다(int8/uint8 인 경우가 많음).
    # 정확한 정합이 필요하면 Furiosa Model Zoo 의 preprocess 를 쓰거나 ONNX 입력 스펙에 맞춰야 한다.
    # 아래는 구조 검증용 float32 NCHW 입력이다.

    cfg = RuntimeConfig(
        backend="warboy",
        engine_path=str(engine_path),
        input_name=args.input_name,
        output_name=args.output_name,
        input_shape=args.input_shape,
        extra={
            "device": args.device,
            "allow_dynamic_shape": args.allow_dynamic_shape,
        },
    )

    rh = create_runtime(cfg)

    x = batch
    try:
        _ = infer(rh, x)  # warmup
    except TypeError as exc:
        if "UINT8" not in str(exc).upper():
            raise
        retried_batch, retried_contexts, retried_kwargs = _maybe_retry_uint8_batch(
            image_path,
            args.input_shape,
            model_helper,
            np,
        )
        if retried_batch is None:
            raise TypeError(
                "Warboy runtime expected UINT8 input for this ENF, but the current example prepared "
                f"dtype={getattr(x, 'dtype', None)!r}. Try the matching vendor branch README or use "
                "a model-zoo preprocess path that yields uint8 input."
            ) from exc
        x = retried_batch
        contexts = retried_contexts
        input_source = f"{image_path} (model-zoo preprocess {retried_kwargs}, UINT8 retry)"
        _ = infer(rh, x)

    times = []
    y = None
    for _ in range(args.iters):
        t0 = timeit.default_timer()
        y = infer(rh, x)
        t1 = timeit.default_timer()
        times.append((t1 - t0) * 1000)

    if isinstance(y, list):
        y = [np.ascontiguousarray(item) for item in y]
    else:
        y = np.ascontiguousarray(y)
    if model_helper is not None and contexts is not None:
        try:
            outputs_for_postprocess = y if isinstance(y, list) else [y]
            result = model_helper.postprocess(outputs_for_postprocess, contexts)
            print(f"postprocess: {result}")
        except Exception as exc:
            print(f"postprocess skipped: {exc!r}")
            if isinstance(y, list):
                print(f"raw_output_shapes: {_format_output_shape(y)}")
            else:
                cls_id = _extract_prediction_id(y, torch)
                if labels and 0 <= cls_id < len(labels):
                    print(f"pred: {labels[cls_id]} (id={cls_id})")
                else:
                    print(f"pred_id: {cls_id} (labels file not found: {labels_path})")
    else:
        if isinstance(y, list):
            print(f"raw_output_shapes: {_format_output_shape(y)}")
        else:
            cls_id = _extract_prediction_id(y, torch)
            if labels and 0 <= cls_id < len(labels):
                print(f"pred: {labels[cls_id]} (id={cls_id})")
            else:
                print(f"pred_id: {cls_id} (labels file not found: {labels_path})")

    print(f"Avg latency: {np.mean(times):.3f} ms, shape={_format_output_shape(y)}")
    print(f"(engine={engine_path}, input={input_source}, device={args.device})")

    destroy_runtime(rh)

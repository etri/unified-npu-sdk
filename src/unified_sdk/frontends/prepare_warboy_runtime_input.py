from __future__ import annotations

import contextlib
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from unified_sdk.frontends.types import PreparedWarboyRuntimeInput


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _center_crop_resize_pil(image, size: int):
    width, height = image.size
    scale = 256 / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)))
    left = max((resized.width - size) // 2, 0)
    top = max((resized.height - size) // 2, 0)
    return resized.crop((left, top, left + size, top + size))


def _normalize_dtype_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, np.dtype):
        return value.name
    dtype = getattr(value, "dtype", None)
    if dtype is not None and dtype is not value:
        return _normalize_dtype_name(dtype)
    text = str(value).strip().lower()
    if not text:
        return None
    if "uint8" in text or text == "u8":
        return "uint8"
    if "float32" in text or "fp32" in text or text == "f32":
        return "float32"
    return text


def _load_image_batch_float32(image_path: Path, input_shape: tuple[int, ...]):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    cropped = _center_crop_resize_pil(image, input_shape[-1])
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    array = array.transpose(2, 0, 1)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
    normalized = (array - mean) / std
    return normalized[None, ...].astype(np.float32)


def _load_image_batch_uint8(image_path: Path, input_shape: tuple[int, ...]):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    cropped = _center_crop_resize_pil(image, input_shape[-1])
    array = np.asarray(cropped, dtype=np.uint8)
    array = array.transpose(2, 0, 1)
    return array[None, ...].astype(np.uint8)


def _normalize_model_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _resolve_model_zoo_target(model_name: str) -> str | None:
    try:
        from furiosa.models import vision
    except Exception:
        return None

    declared = getattr(vision, "__all__", None)
    if isinstance(declared, (list, tuple)):
        candidates = [str(name) for name in declared if isinstance(name, str) and not name.startswith("_")]
    else:
        candidates = [name for name in dir(vision) if not name.startswith("_")]

    normalized = _normalize_model_name(model_name)
    for candidate in sorted(candidates):
        if _normalize_model_name(candidate) == normalized:
            return candidate
    return None


def _maybe_create_model_zoo_helper(engine_path: Path):
    try:
        from furiosa.models import vision
    except Exception:
        return None, "furiosa.models.vision import failed"

    resolved = _resolve_model_zoo_target(engine_path.stem)
    if resolved is None or not hasattr(vision, resolved):
        return None, f"no matching model-zoo helper for {engine_path.stem}"
    try:
        return getattr(vision, resolved)(), None
    except Exception as exc:
        return None, f"failed to construct model-zoo helper {resolved!r}: {exc!r}"


def _extract_first_input(inputs: Any):
    if isinstance(inputs, (list, tuple)) and len(inputs) == 1:
        return inputs[0]
    return inputs


def inspect_warboy_input_contract(engine_path: Path, *, device: str | None = None) -> dict[str, Any]:
    try:
        from furiosa.runtime import sync
    except Exception:
        return {"expected_dtype": None, "inspection_warning": "furiosa.runtime unavailable"}

    runner = None
    try:
        try:
            runner = sync.create_runner(str(engine_path), device=str(device)) if device else sync.create_runner(str(engine_path))
        except TypeError:
            runner = sync.create_runner(str(engine_path))

        inputs = None
        for attr in ("inputs", "input_specs", "input_tensors"):
            if hasattr(runner, attr):
                value = getattr(runner, attr)
                try:
                    value = value() if callable(value) else value
                except Exception:
                    continue
                if value:
                    inputs = value
                    break

        expected_dtype = None
        if isinstance(inputs, (list, tuple)) and inputs:
            first = inputs[0]
            for attr in ("dtype", "data_type", "type"):
                expected_dtype = _normalize_dtype_name(getattr(first, attr, None))
                if expected_dtype:
                    break

        return {"expected_dtype": expected_dtype, "inspection_warning": None}
    except Exception as exc:
        return {"expected_dtype": None, "inspection_warning": f"input contract inspect failed: {exc!r}"}
    finally:
        if runner is not None:
            close = getattr(runner, "close", None)
            with contextlib.suppress(Exception):
                close() if callable(close) else None


def _load_with_model_zoo_preprocess(model_helper, image_path: Path, expected_dtype: str | None):
    preprocess_error = None
    kwargs_candidates = ({}, {"with_scaling": True}) if expected_dtype == "uint8" else ({"with_scaling": True}, {})
    best_candidate = None

    for kwargs in kwargs_candidates:
        for candidate in ([str(image_path)], str(image_path)):
            try:
                inputs, contexts = model_helper.preprocess(candidate, **kwargs)
                arr = _extract_first_input(inputs)
                actual_dtype = _normalize_dtype_name(arr)
                if expected_dtype is not None and actual_dtype != expected_dtype:
                    best_candidate = best_candidate or (inputs, contexts, kwargs, actual_dtype)
                    continue
                return inputs, contexts, kwargs, actual_dtype
            except Exception as exc:
                preprocess_error = exc

    if best_candidate is not None:
        inputs, contexts, kwargs, actual_dtype = best_candidate
        return inputs, contexts, kwargs, actual_dtype
    raise RuntimeError(f"Model Zoo preprocess failed: {preprocess_error!r}")


def _load_synthetic_with_model_zoo_preprocess(model_helper, input_shape: tuple[int, ...], expected_dtype: str | None):
    from PIL import Image

    size = input_shape[-1]
    synthetic = Image.new("RGB", (size, size), color=(127, 127, 127))
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        synthetic.save(tmp_path, format="JPEG")
        return _load_with_model_zoo_preprocess(model_helper, tmp_path, expected_dtype)
    finally:
        if tmp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()


def prepare_warboy_runtime_input(
    *,
    engine_path: str | Path,
    image_path: str | Path,
    input_shape: tuple[int, ...],
    device: str | None = None,
) -> PreparedWarboyRuntimeInput:
    engine = Path(engine_path).expanduser().resolve()
    image = Path(image_path).expanduser().resolve()

    contract = inspect_warboy_input_contract(engine, device=device)
    expected_dtype = contract.get("expected_dtype")
    warnings: list[str] = []
    if contract.get("inspection_warning"):
        warnings.append(str(contract["inspection_warning"]))

    model_helper, helper_warning = _maybe_create_model_zoo_helper(engine)
    if helper_warning is not None:
        warnings.append(helper_warning)

    if image.is_file():
        if model_helper is not None:
            inputs, contexts, preprocess_kwargs, actual_dtype = _load_with_model_zoo_preprocess(
                model_helper,
                image,
                expected_dtype,
            )
            return PreparedWarboyRuntimeInput(
                batch=_extract_first_input(inputs),
                contexts=contexts,
                model_helper=model_helper,
                source_description=f"{image} (model-zoo preprocess {preprocess_kwargs})",
                expected_dtype=expected_dtype,
                actual_dtype=actual_dtype,
                warnings=tuple(warnings),
            )

        if expected_dtype == "uint8":
            batch = _load_image_batch_uint8(image, input_shape)
            actual_dtype = "uint8"
            source_description = f"{image} (generic uint8 image fallback)"
        else:
            batch = _load_image_batch_float32(image, input_shape)
            actual_dtype = "float32"
            source_description = f"{image} (generic float32 image fallback)"

        return PreparedWarboyRuntimeInput(
            batch=batch,
            contexts=None,
            model_helper=None,
            source_description=source_description,
            expected_dtype=expected_dtype,
            actual_dtype=actual_dtype,
            warnings=tuple(warnings),
        )

    if model_helper is not None:
        try:
            inputs, contexts, preprocess_kwargs, actual_dtype = _load_synthetic_with_model_zoo_preprocess(
                model_helper,
                input_shape,
                expected_dtype,
            )
            return PreparedWarboyRuntimeInput(
                batch=_extract_first_input(inputs),
                contexts=contexts,
                model_helper=model_helper,
                source_description=f"synthetic RGB image via model-zoo preprocess {preprocess_kwargs}",
                expected_dtype=expected_dtype,
                actual_dtype=actual_dtype,
                warnings=tuple(warnings),
            )
        except Exception as exc:
            warnings.append(f"model-zoo synthetic preprocess failed: {exc!r}")

    if expected_dtype == "uint8":
        batch = np.zeros(input_shape, dtype=np.uint8)
        actual_dtype = "uint8"
        source_description = f"synthetic zeros uint8 {input_shape} (generic fallback)"
    else:
        batch = np.zeros(input_shape, dtype=np.float32)
        actual_dtype = "float32"
        source_description = f"synthetic zeros float32 {input_shape} (generic fallback)"

    return PreparedWarboyRuntimeInput(
        batch=batch,
        contexts=None,
        model_helper=None,
        source_description=source_description,
        expected_dtype=expected_dtype,
        actual_dtype=actual_dtype,
        warnings=tuple(warnings),
    )

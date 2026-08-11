from __future__ import annotations

import contextlib
import json
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
    if text == "float" or "float32" in text or "fp32" in text or text == "f32":
        return "float32"
    return text


def _infer_dtype_from_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    upper = text.upper()
    if "UINT8" in upper or "U8" == upper.strip():
        return "uint8"
    if "FLOAT32" in upper or "FP32" in upper or "F32" == upper.strip():
        return "float32"
    return None


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


def _metadata_sidecar_path(enf_path: Path) -> Path:
    return Path(f"{enf_path}.json")


def _read_sidecar_input_contract(engine_path: Path) -> dict[str, Any] | None:
    sidecar = _metadata_sidecar_path(engine_path)
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text())
    except Exception as exc:
        return {
            "expected_dtype": None,
            "input_shape": None,
            "inspection_warning": f"sidecar metadata parse failed: {exc!r}",
            "contract_source": "sidecar",
        }
    if not isinstance(payload, dict):
        return {
            "expected_dtype": None,
            "input_shape": None,
            "inspection_warning": "sidecar metadata is not a JSON object",
            "contract_source": "sidecar",
        }
    contract = payload.get("input_contract")
    if not isinstance(contract, dict):
        return {
            "expected_dtype": None,
            "input_shape": None,
            "inspection_warning": "sidecar metadata does not include input_contract",
            "contract_source": "sidecar",
        }
    return {
        "expected_dtype": _normalize_dtype_name(contract.get("input_dtype")),
        "input_shape": contract.get("input_shape"),
        "inspection_warning": contract.get("inspection_warning"),
        "contract_source": "sidecar",
    }


def _extract_first_input(inputs: Any):
    if isinstance(inputs, (list, tuple)) and len(inputs) == 1:
        return inputs[0]
    return inputs


def inspect_warboy_input_contract(engine_path: Path, *, device: str | None = None) -> dict[str, Any]:
    sidecar_contract = _read_sidecar_input_contract(engine_path)
    if sidecar_contract is not None and sidecar_contract.get("expected_dtype") is not None:
        return sidecar_contract

    try:
        from furiosa.runtime import sync
    except Exception:
        warning = "furiosa.runtime unavailable"
        if sidecar_contract is not None and sidecar_contract.get("inspection_warning"):
            warning = f"{sidecar_contract['inspection_warning']}; {warning}"
        return {
            "expected_dtype": None,
            "inspection_warning": warning,
            "contract_source": "runtime_inspect",
        }

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
            if expected_dtype is None:
                expected_dtype = _infer_dtype_from_text(first)
            if expected_dtype is None:
                for attr in ("name", "desc", "description", "format"):
                    expected_dtype = _infer_dtype_from_text(getattr(first, attr, None))
                    if expected_dtype:
                        break

        warning = None
        if expected_dtype is None and sidecar_contract is not None:
            warning = sidecar_contract.get("inspection_warning")
        return {
            "expected_dtype": expected_dtype,
            "inspection_warning": warning,
            "contract_source": "runtime_inspect",
        }
    except Exception as exc:
        warning = f"input contract inspect failed: {exc!r}"
        if sidecar_contract is not None and sidecar_contract.get("inspection_warning"):
            warning = f"{sidecar_contract['inspection_warning']}; {warning}"
        return {
            "expected_dtype": None,
            "inspection_warning": warning,
            "contract_source": "runtime_inspect",
        }
    finally:
        if runner is not None:
            close = getattr(runner, "close", None)
            with contextlib.suppress(Exception):
                close() if callable(close) else None


def _run_model_zoo_preprocess_candidates(model_helper, image_path: Path):
    results = []
    preprocess_error = None
    for kwargs in ({"with_scaling": True}, {}):
        for candidate in ([str(image_path)], str(image_path)):
            try:
                inputs, contexts = model_helper.preprocess(candidate, **kwargs)
                arr = _extract_first_input(inputs)
                results.append((inputs, contexts, kwargs, _normalize_dtype_name(arr)))
                break
            except Exception as exc:
                preprocess_error = exc
    if not results:
        raise RuntimeError(f"Model Zoo preprocess failed: {preprocess_error!r}")
    return results


def _load_with_model_zoo_preprocess(model_helper, image_path: Path, expected_dtype: str | None):
    results = _run_model_zoo_preprocess_candidates(model_helper, image_path)
    if expected_dtype is not None:
        for inputs, contexts, kwargs, actual_dtype in results:
            if actual_dtype == expected_dtype:
                return inputs, contexts, kwargs, actual_dtype
        available = sorted({dtype or "unknown" for _, _, _, dtype in results})
        raise RuntimeError(
            f"Model Zoo preprocess did not yield expected dtype={expected_dtype!r}; "
            f"available candidates={available}"
        )

    available_dtypes = sorted({dtype or "unknown" for _, _, _, dtype in results})
    if len(available_dtypes) == 1:
        inputs, contexts, kwargs, actual_dtype = results[0]
        return inputs, contexts, kwargs, actual_dtype

    raise RuntimeError(
        "Unable to resolve Warboy input dtype automatically; "
        f"model-zoo preprocess yielded multiple dtype candidates={available_dtypes}. "
        "Re-run with an explicit input dtype override."
    )


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
    preferred_dtype: str | None = None,
) -> PreparedWarboyRuntimeInput:
    engine = Path(engine_path).expanduser().resolve()
    image = Path(image_path).expanduser().resolve()

    contract = inspect_warboy_input_contract(engine, device=device)
    expected_dtype = preferred_dtype or contract.get("expected_dtype")
    warnings: list[str] = []
    if contract.get("inspection_warning"):
        warnings.append(str(contract["inspection_warning"]))
    if preferred_dtype is not None:
        warnings.append(f"using explicit input dtype override: {preferred_dtype}")
    elif expected_dtype is None:
        warnings.append("input contract dtype could not be resolved automatically")

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
        elif expected_dtype == "float32":
            batch = _load_image_batch_float32(image, input_shape)
            actual_dtype = "float32"
            source_description = f"{image} (generic float32 image fallback)"
        else:
            raise RuntimeError(
                "Unable to determine Warboy runtime input dtype for generic image fallback. "
                "Re-run with --input-dtype uint8 or --input-dtype float32."
            )

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
    elif expected_dtype == "float32":
        batch = np.zeros(input_shape, dtype=np.float32)
        actual_dtype = "float32"
        source_description = f"synthetic zeros float32 {input_shape} (generic fallback)"
    else:
        raise RuntimeError(
            "Unable to determine Warboy runtime input dtype for synthetic fallback. "
            "Re-run with --input-dtype uint8 or --input-dtype float32."
        )

    return PreparedWarboyRuntimeInput(
        batch=batch,
        contexts=None,
        model_helper=None,
        source_description=source_description,
        expected_dtype=expected_dtype,
        actual_dtype=actual_dtype,
        warnings=tuple(warnings),
    )

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

from unified_sdk.runtime.registry import register
from unified_sdk.types import RuntimeConfig, RuntimeHandle


_CAPABILITY_FAMILY = "vision.direct-python-runtime"
_RUNTIME_PIPELINE = (
    "validate_runtime_config",
    "load_vendor_runtime",
    "validate_input",
    "run_vendor_inference",
    "normalize_output",
    "destroy_runtime",
)
_VENDOR_API_MAP = {
    "create_runtime": "rebel.Runtime(str(path), device=..., tensor_type=..., activate_profiler=..., timeout=...)",
    "infer": "runtime(input_array)",
    "destroy": "clear RuntimeHandle.ctx",
}

_TENSOR_TYPES = {"np", "pt"}


def _require_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"RuntimeConfig.{field_name} must be a non-empty string")
    return value.strip()


def _validate_shape(shape: Tuple[int, ...], field_name: str) -> Tuple[int, ...]:
    if not isinstance(shape, tuple) or not shape:
        raise ValueError(f"RuntimeConfig.{field_name} must be a non-empty tuple of positive integers")
    if not all(isinstance(dim, int) and dim > 0 for dim in shape):
        raise ValueError(f"RuntimeConfig.{field_name} must contain only positive integers: {shape!r}")
    return shape


def _parse_device(value: Any) -> int:
    try:
        device = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"RuntimeConfig.extra['device'] must be an integer, got {value!r}") from exc
    if device < 0:
        raise ValueError("RuntimeConfig.extra['device'] must be >= 0")
    return device


def _parse_tensor_type(value: Any) -> str:
    tensor_type = str(value)
    if tensor_type not in _TENSOR_TYPES:
        raise ValueError(
            f"RuntimeConfig.extra['tensor_type'] must be one of {sorted(_TENSOR_TYPES)}, got {value!r}"
        )
    return tensor_type


def _parse_timeout(value: Any) -> Any:
    if value is None:
        return None
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"RuntimeConfig.extra['timeout'] must be numeric, got {value!r}") from exc
    if timeout <= 0:
        raise ValueError("RuntimeConfig.extra['timeout'] must be > 0")
    return timeout


def _parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    if value in (0, 1):
        return bool(value)
    raise ValueError(f"RuntimeConfig.extra['{field_name}'] must be a boolean-like value, got {value!r}")


def _to_numpy(output: Any) -> np.ndarray:
    if isinstance(output, np.ndarray):
        return output
    if hasattr(output, "detach") and callable(output.detach):
        return output.detach().cpu().numpy()
    if isinstance(output, (list, tuple)):
        if len(output) != 1:
            raise TypeError(
                "RBLN runtime returned multiple outputs; pass a model with one output or handle raw output directly"
            )
        return _to_numpy(output[0])
    return np.asarray(output)


class _RBLNRuntime:
    name = "rbln"

    def create(self, cfg: RuntimeConfig) -> RuntimeHandle:
        if cfg.backend != self.name:
            raise ValueError(f"RBLN runtime adapter received backend={cfg.backend!r}")

        p = Path(cfg.engine_path)
        if not p.exists():
            raise FileNotFoundError(f"RBLN model not found: {p}")
        if p.suffix != ".rbln":
            raise ValueError(f"Expected a .rbln model file, got: {p}")

        input_name = _require_non_empty_string(cfg.input_name, "input_name")
        output_name = _require_non_empty_string(cfg.output_name, "output_name")
        input_shape = _validate_shape(tuple(cfg.input_shape), "input_shape")

        extra = dict(cfg.extra or {})
        device = _parse_device(extra.get("device", 0))
        tensor_type = _parse_tensor_type(extra.get("tensor_type", "np"))
        timeout = _parse_timeout(extra.get("timeout", None))
        activate_profiler = _parse_bool(extra.get("activate_profiler", False), "activate_profiler")

        import rebel

        try:
            runtime = rebel.Runtime(
                str(p),
                device=device,
                tensor_type=tensor_type,
                activate_profiler=activate_profiler,
                timeout=timeout,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to create RBLN runtime for {p}: {exc}") from exc

        return RuntimeHandle(
            backend=self.name,
            engine_path=str(p),
            input_name=input_name,
            output_name=output_name,
            input_shape=input_shape,
            ctx={
                "runtime": runtime,
                "tensor_type": tensor_type,
                "device": device,
                "extra": extra,
                "capability_family": _CAPABILITY_FAMILY,
                "runtime_pipeline": _RUNTIME_PIPELINE,
                "vendor_api_map": _VENDOR_API_MAP,
            },
        )

    def infer(self, rh: RuntimeHandle, input_array: np.ndarray) -> np.ndarray:
        if not rh.ctx or "runtime" not in rh.ctx:
            raise RuntimeError("RBLN RuntimeHandle is closed or invalid")

        rt = rh.ctx["runtime"]
        extra = rh.ctx.get("extra", {})
        allow_dynamic = _parse_bool(extra.get("allow_dynamic_shape", False), "allow_dynamic_shape")

        input_shape = tuple(getattr(input_array, "shape", ()))
        if (not allow_dynamic) and input_shape != tuple(rh.input_shape):
            raise ValueError(
                f"Bad input shape: {getattr(input_array, 'shape', None)}, expected {rh.input_shape}"
            )

        try:
            out = rt(input_array)
        except Exception as exc:
            raise RuntimeError(f"RBLN inference failed: {exc}") from exc

        return _to_numpy(out)

    def destroy(self, rh: RuntimeHandle) -> None:
        rh.ctx.clear()


register(_RBLNRuntime())

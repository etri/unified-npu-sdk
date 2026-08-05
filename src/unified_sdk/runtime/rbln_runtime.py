from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np

from unified_sdk.options import resolve_rbln_vision_runtime_options
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
_VENDOR_TO_UNIFIED_API_MAP = {
    "rebel.Runtime(str(path), ...)": "create_runtime(cfg)",
    "runtime(input_array)": "infer(rh, input_array)",
    "runtime output tensor/list": "infer(...) return np.ndarray",
    "RuntimeHandle.ctx.clear()": "destroy_runtime(rh)",
}


def describe_api_mapping() -> Dict[str, Any]:
    return {
        "unified_api": {
            "create": "create_runtime(cfg)",
            "infer": "infer(rh, input_array)",
            "destroy": "destroy_runtime(rh)",
        },
        "backend": "rbln",
        "capability_family": _CAPABILITY_FAMILY,
        "mapping_direction": "vendor_api ==> unified_api",
        "pipeline": _RUNTIME_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "vendor_to_unified_api_map": _VENDOR_TO_UNIFIED_API_MAP,
    }


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

        options = resolve_rbln_vision_runtime_options(cfg.backend_options, extra=dict(cfg.extra or {}))
        options_meta = options.to_metadata()

        import rebel

        try:
            runtime = rebel.Runtime(
                str(p),
                device=options.device,
                tensor_type=options.tensor_type,
                activate_profiler=options.activate_profiler,
                timeout=options.timeout,
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
                "backend_options": options_meta,
                "capability_family": _CAPABILITY_FAMILY,
                "runtime_pipeline": _RUNTIME_PIPELINE,
                "vendor_api_map": _VENDOR_API_MAP,
            },
        )

    def infer(self, rh: RuntimeHandle, input_array: np.ndarray) -> np.ndarray:
        if not rh.ctx or "runtime" not in rh.ctx:
            raise RuntimeError("RBLN RuntimeHandle is closed or invalid")

        rt = rh.ctx["runtime"]
        options_meta = dict(rh.ctx.get("backend_options", {}))
        allow_dynamic = bool(options_meta.get("allow_dynamic_shape", False))

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

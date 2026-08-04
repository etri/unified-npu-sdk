from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from unified_sdk.options import resolve_qb_runtime_options
from unified_sdk.runtime._qb_common import (
    build_model_config,
    load_qbruntime_modules,
    require_non_empty_string,
    to_numpy,
    validate_mxq_path,
    validate_shape,
)
from unified_sdk.runtime.registry import register
from unified_sdk.types import RuntimeConfig, RuntimeHandle


_CAPABILITY_FAMILY = "vision.direct-python-runtime"
_RUNTIME_PIPELINE = (
    "validate_runtime_config",
    "resolve_model_config",
    "load_vendor_model",
    "validate_input",
    "run_vendor_inference",
    "normalize_output",
    "destroy_runtime",
)
_VENDOR_API_MAP = {
    "model_config": "qbruntime.type.ModelConfig / qbruntime.type.CoreMode",
    "create_runtime": "qbruntime.model.load(str(mxq_path), model_config)",
    "infer": "model.infer([input_array])",
    "destroy": "model.dispose/release/unload/close best-effort",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "qbruntime.type.ModelConfig / CoreMode": "QBVisionRuntimeOptions.core_mode",
    "qbruntime.model.load(str(mxq_path), model_config)": "create_runtime(cfg)",
    "model.infer([input_array])": "infer(rh, input_array)",
    "qbruntime output tensor/list": "infer(...) return np.ndarray or list[np.ndarray]",
    "model.dispose/release/unload/close": "destroy_runtime(rh)",
}


def _legacy_fallback_metadata(cfg: RuntimeConfig) -> dict[str, Any]:
    used = cfg.backend_options is None and bool(cfg.extra)
    return {
        "legacy_extra_fallback_used": used,
        "legacy_extra_keys": sorted(dict(cfg.extra or {}).keys()) if used else [],
    }


def describe_api_mapping() -> dict[str, Any]:
    return {
        "unified_api": {
            "create": "create_runtime(cfg)",
            "infer": "infer(rh, input_array)",
            "destroy": "destroy_runtime(rh)",
        },
        "backend": "qb",
        "capability_family": _CAPABILITY_FAMILY,
        "mapping_direction": "vendor_api ==> unified_api",
        "pipeline": _RUNTIME_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "vendor_to_unified_api_map": _VENDOR_TO_UNIFIED_API_MAP,
    }


class _QBVisionRuntime:
    """Mobilint ARISE(QB) vision runtime adapter."""

    name = "qb"

    def create(self, cfg: RuntimeConfig) -> RuntimeHandle:
        if cfg.backend != self.name:
            raise ValueError(f"QB runtime adapter received backend={cfg.backend!r}")

        p = validate_mxq_path(cfg.engine_path)
        input_name = require_non_empty_string(cfg.input_name, "RuntimeConfig.input_name")
        output_name = require_non_empty_string(cfg.output_name, "RuntimeConfig.output_name")
        input_shape = validate_shape(tuple(cfg.input_shape), "RuntimeConfig.input_shape")

        options = resolve_qb_runtime_options(cfg.backend_options, cfg.extra)
        core_mode = options.core_mode

        qbruntime, qb_model, qb_type = load_qbruntime_modules()
        model_config = build_model_config(qb_type, core_mode)
        try:
            model = qb_model.load(str(p), model_config)
        except Exception as exc:
            detail = str(exc)
            if "CoreMode::Auto" in detail:
                detail += (
                    " Multi-core-mode MXQ cannot be loaded with core_mode=auto. "
                    "Pass an explicit core mode such as 'single', 'global4', or 'global8'."
                )
            raise RuntimeError(f"Failed to load QB model for {p}: {detail}") from exc

        return RuntimeHandle(
            backend=self.name,
            engine_path=str(p),
            input_name=input_name,
            output_name=output_name,
            input_shape=input_shape,
            ctx={
                "model": model,
                "qbruntime": qbruntime,
                "core_mode": core_mode,
                "runtime_options": options,
                **_legacy_fallback_metadata(cfg),
                "capability_family": _CAPABILITY_FAMILY,
                "runtime_pipeline": _RUNTIME_PIPELINE,
                "vendor_api_map": _VENDOR_API_MAP,
            },
        )

    def infer(self, rh: RuntimeHandle, input_array: np.ndarray) -> Any:
        if not rh.ctx or "model" not in rh.ctx:
            raise RuntimeError("QB RuntimeHandle is closed or invalid")

        model = rh.ctx["model"]
        runtime_options = rh.ctx.get("runtime_options")
        allow_dynamic = bool(runtime_options.allow_dynamic_shape) if runtime_options is not None else False

        input_shape = tuple(getattr(input_array, "shape", ()))
        if (not allow_dynamic) and input_shape != tuple(rh.input_shape):
            raise ValueError(
                f"Bad input shape: {getattr(input_array, 'shape', None)}, expected {rh.input_shape}"
            )

        try:
            out = model.infer([input_array])
        except Exception as exc:
            raise RuntimeError(f"QB inference failed: {exc}") from exc

        return to_numpy(out)

    def destroy(self, rh: RuntimeHandle) -> None:
        model = rh.ctx.get("model") if rh.ctx else None
        if model is not None:
            for method in ("dispose", "release", "unload", "close"):
                fn = getattr(model, method, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
                    break
        rh.ctx.clear()


register(_QBVisionRuntime())

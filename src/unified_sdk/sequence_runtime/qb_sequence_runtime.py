from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from unified_sdk.options import resolve_qb_sequence_runtime_options
from unified_sdk.runtime._qb_common import (
    build_model_config,
    load_qbruntime_modules,
    parse_non_negative_int,
    require_non_empty_string,
    to_numpy,
    validate_mxq_path,
    validate_shape,
)
from unified_sdk.sequence_runtime._qb_sequence_common import normalize_batch_params
from unified_sdk.sequence_runtime.registry import register
from unified_sdk.sequence_runtime.types import (
    SequenceBatchParam,
    SequenceRuntimeConfig,
    SequenceRuntimeHandle,
)


_CAPABILITY_FAMILY = "sequence.low-level-cache-aware-runtime"
_RUNTIME_PIPELINE = (
    "validate_runtime_config",
    "resolve_model_config",
    "load_vendor_model",
    "validate_input",
    "normalize_batch_metadata",
    "run_vendor_inference",
    "normalize_output",
    "destroy_runtime",
)
_VENDOR_API_MAP = {
    "model_config": "qbruntime.type.ModelConfig / qbruntime.type.CoreMode",
    "create_runtime": "qbruntime.model.load(str(mxq_path), model_config)",
    "infer": "model.infer([input_array], cache_size=..., params=...)",
    "batch_param": "qbruntime.BatchParam(sequence_length, cache_size, cache_id)",
    "destroy": "model.dispose/release/unload/close best-effort",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "qbruntime.type.ModelConfig / CoreMode": "QBSequenceRuntimeOptions.core_mode",
    "qbruntime.model.load(str(mxq_path), model_config)": "create_sequence_runtime(cfg)",
    "qbruntime.BatchParam(...)": "SequenceBatchParam(sequence_length, cache_size, cache_id)",
    "model.infer([input_array], cache_size=..., params=...)": "infer_sequence(rh, input_array, cache_size=..., batch_params=...)",
    "model.dispose/release/unload/close": "destroy_sequence_runtime(rh)",
}


def describe_api_mapping() -> dict[str, Any]:
    return {
        "unified_api": {
            "create": "create_sequence_runtime(cfg)",
            "infer": "infer_sequence(rh, input_array, cache_size=..., batch_params=...)",
            "destroy": "destroy_sequence_runtime(rh)",
        },
        "backend": "qb",
        "capability_family": _CAPABILITY_FAMILY,
        "mapping_direction": "vendor_api ==> unified_api",
        "pipeline": _RUNTIME_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "vendor_to_unified_api_map": _VENDOR_TO_UNIFIED_API_MAP,
    }


class _QBSequenceRuntime:
    """Mobilint ARISE(QB) low-level sequence extension runtime adapter."""

    name = "qb"

    def create(self, cfg: SequenceRuntimeConfig) -> SequenceRuntimeHandle:
        if cfg.backend != self.name:
            raise ValueError(f"QB sequence runtime adapter received backend={cfg.backend!r}")

        p = validate_mxq_path(cfg.engine_path)
        input_name = require_non_empty_string(cfg.input_name, "SequenceRuntimeConfig.input_name")
        output_name = require_non_empty_string(cfg.output_name, "SequenceRuntimeConfig.output_name")
        input_shape = validate_shape(tuple(cfg.input_shape), "SequenceRuntimeConfig.input_shape")

        options = resolve_qb_sequence_runtime_options(cfg.backend_options)
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
            raise RuntimeError(f"Failed to load QB sequence model for {p}: {detail}") from exc

        return SequenceRuntimeHandle(
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
                "capability_family": _CAPABILITY_FAMILY,
                "runtime_pipeline": _RUNTIME_PIPELINE,
                "vendor_api_map": _VENDOR_API_MAP,
            },
        )

    def infer(
        self,
        rh: SequenceRuntimeHandle,
        input_array: np.ndarray,
        *,
        cache_size: int = 0,
        batch_params: Optional[Sequence[SequenceBatchParam]] = None,
    ) -> Any:
        if not rh.ctx or "model" not in rh.ctx:
            raise RuntimeError("QB SequenceRuntimeHandle is closed or invalid")

        model = rh.ctx["model"]
        qbruntime_module = rh.ctx.get("qbruntime")
        runtime_options = rh.ctx.get("runtime_options")
        allow_dynamic = bool(runtime_options.allow_dynamic_shape) if runtime_options is not None else False
        cache_size = parse_non_negative_int(cache_size, "cache_size")
        normalized_batch_params = normalize_batch_params(batch_params, qbruntime_module)

        input_shape = tuple(getattr(input_array, "shape", ()))
        if (not allow_dynamic) and input_shape != tuple(rh.input_shape):
            raise ValueError(
                f"Bad input shape: {getattr(input_array, 'shape', None)}, expected {rh.input_shape}"
            )

        try:
            if normalized_batch_params is not None:
                out = model.infer([input_array], params=normalized_batch_params)
            elif cache_size > 0:
                out = model.infer([input_array], cache_size=cache_size)
            else:
                out = model.infer([input_array])
        except Exception as exc:
            raise RuntimeError(f"QB sequence inference failed: {exc}") from exc

        return to_numpy(out)

    def destroy(self, rh: SequenceRuntimeHandle) -> None:
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


register(_QBSequenceRuntime())

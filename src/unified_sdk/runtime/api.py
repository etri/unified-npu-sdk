from __future__ import annotations

from typing import Any, Dict

import numpy as np

from unified_sdk.runtime.registry import get_llm_runtime, get_runtime
from unified_sdk.types import LLMRuntimeConfig, LLMRuntimeHandle, RuntimeConfig, RuntimeHandle

import warnings

try:
    from . import qb_runtime as _qb  # noqa: F401
except Exception as e:
    warnings.warn(f"QB backend disabled: {e!r}")
    _qb = None

try:
    from . import tensorrt_runtime as _tensorrt  # noqa: F401
except Exception as e:
    warnings.warn(f"TensorRT backend disabled: {e!r}")
    _tensorrt = None

try:
    from . import rbln_runtime as _rbln  # noqa: F401
except Exception as e:
    warnings.warn(f"RBLN backend disabled: {e!r}")
    _rbln = None

try:
    from . import warboy_runtime as _warboy  # noqa: F401
except Exception as e:
    warnings.warn(f"Warboy backend disabled: {e!r}")
    _warboy = None

try:
    from . import rngd_runtime as _rngd  # noqa: F401
except Exception as e:
    warnings.warn(f"RNGD backend disabled: {e!r}")
    _rngd = None

try:
    from . import rbln_llm_runtime as _rbln_llm  # noqa: F401
except Exception as e:
    warnings.warn(f"RBLN LLM runtime mapping disabled: {e!r}")
    _rbln_llm = None

try:
    from . import tensorrt_llm_runtime as _tensorrt_llm  # noqa: F401
except Exception as e:
    warnings.warn(f"TensorRT-LLM runtime mapping disabled: {e!r}")
    _tensorrt_llm = None


def _resolve_backend_name(target: Any) -> str:
    backend = getattr(target, "backend", target)
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError(
            "A backend name or object with a non-empty .backend field is required. "
            "Examples: 'tensorrt', 'rbln', 'qb', 'warboy', 'rngd'."
        )
    return backend.strip()


def create_runtime(cfg: RuntimeConfig) -> RuntimeHandle:
    adapter = get_runtime(cfg.backend)
    return adapter.create(cfg)


def infer(rh: RuntimeHandle, input_array: "np.ndarray") -> "np.ndarray":
    adapter = get_runtime(rh.backend)
    return adapter.infer(rh, input_array)


def destroy_runtime(rh: RuntimeHandle) -> None:
    adapter = get_runtime(rh.backend)
    return adapter.destroy(rh)


def create_runtime_LLM(cfg: LLMRuntimeConfig) -> Any:
    adapter = get_llm_runtime(cfg.backend)
    return adapter.create(cfg)


def infer_LLM(rh: Any, input_or_prompt: Any, **kwargs: Any) -> Any:
    adapter = get_llm_runtime(rh.backend)
    return adapter.infer(rh, input_or_prompt, **kwargs)


def generate_LLM(rh: Any, prompt: Any, **overrides: Any) -> Any:
    adapter = get_llm_runtime(rh.backend)
    return adapter.infer(rh, prompt, **overrides)


def destroy_runtime_LLM(rh: Any) -> None:
    adapter = get_llm_runtime(rh.backend)
    return adapter.destroy(rh)


def describe_runtime_api_mapping(target: Any) -> Dict[str, Any]:
    backend = _resolve_backend_name(target)
    if backend == "qb":
        if _qb is None:
            raise RuntimeError("QB runtime backend is unavailable in this environment.")
        return _qb.describe_api_mapping()
    if backend == "tensorrt":
        if _tensorrt is None:
            raise RuntimeError("TensorRT runtime backend is unavailable in this environment.")
        return _tensorrt.describe_api_mapping()
    if backend == "rbln":
        if _rbln is None:
            raise RuntimeError("RBLN runtime backend is unavailable in this environment.")
        return _rbln.describe_api_mapping()
    if backend == "warboy":
        if _warboy is None:
            raise RuntimeError("Warboy runtime backend is unavailable in this environment.")
        return _warboy.describe_api_mapping()
    if backend == "rngd":
        if _rngd is None:
            raise RuntimeError("RNGD runtime backend is unavailable in this environment.")
        return _rngd.describe_api_mapping()
    raise ValueError(f"Unsupported runtime mapping backend: {backend!r}")


def describe_runtime_api_mapping_LLM(target: Any) -> Dict[str, Any]:
    backend = _resolve_backend_name(target)
    if backend == "rbln":
        if _rbln_llm is None:
            raise RuntimeError("RBLN LLM runtime backend is unavailable in this environment.")
        return _rbln_llm.describe_api_mapping()
    if backend == "tensorrt":
        if _tensorrt_llm is None:
            raise RuntimeError("TensorRT-LLM runtime backend is unavailable in this environment.")
        return _tensorrt_llm.describe_api_mapping()
    if backend == "rngd":
        if _rngd is None:
            raise RuntimeError("RNGD runtime backend is unavailable in this environment.")
        return _rngd.describe_api_mapping()
    raise ValueError(f"Unsupported LLM runtime mapping backend: {backend!r}")

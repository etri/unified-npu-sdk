from __future__ import annotations
from typing import Any

import numpy as np

from unified_sdk.runtime.registry import get_runtime
from unified_sdk.types import LLMRuntimeConfig, LLMRuntimeHandle, RuntimeConfig, RuntimeHandle, RuntimeBackendName

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
    from .rbln_llm_runtime import create_llm as _create_rbln_llm, generate_llm as _generate_rbln_llm, destroy_llm as _destroy_rbln_llm
except Exception as e:
    warnings.warn(f"RBLN LLM runtime disabled: {e!r}")
    _create_rbln_llm = _generate_rbln_llm = _destroy_rbln_llm = None

try:
    from .tensorrt_llm_runtime import create_llm as _create_trt_llm, generate_llm as _generate_trt_llm, destroy_llm as _destroy_trt_llm
except Exception as e:
    warnings.warn(f"TensorRT-LLM runtime disabled: {e!r}")
    _create_trt_llm = _generate_trt_llm = _destroy_trt_llm = None


def create_runtime(cfg: RuntimeConfig) -> RuntimeHandle:
    adapter = get_runtime(cfg.backend)
    return adapter.create(cfg)

def infer(rh: RuntimeHandle, input_array: "np.ndarray") -> "np.ndarray":
    adapter = get_runtime(rh.backend)
    return adapter.infer(rh, input_array)

def destroy_runtime(rh: RuntimeHandle) -> None:
    adapter = get_runtime(rh.backend)
    return adapter.destroy(rh)


def create_runtime_LLM(cfg: Any) -> Any:
    backend = getattr(cfg, "backend", None)
    if backend == "rbln":
        if _create_rbln_llm is None:
            raise RuntimeError("RBLN LLM runtime is unavailable in this environment.")
        return _create_rbln_llm(cfg)  # type: ignore[arg-type]
    if backend == "tensorrt":
        if _create_trt_llm is None:
            raise RuntimeError("TensorRT-LLM runtime is unavailable in this environment.")
        return _create_trt_llm(cfg)  # type: ignore[arg-type]

    adapter = get_runtime(backend)
    create_llm = getattr(adapter, "create_llm", None)
    if callable(create_llm):
        return create_llm(cfg)
    return adapter.create(cfg)  # type: ignore[arg-type]


def infer_LLM(rh: Any, input_or_prompt: Any, **kwargs: Any) -> Any:
    backend = getattr(rh, "backend", None)
    if backend in {"rbln", "tensorrt"}:
        return generate_LLM(rh, input_or_prompt, **kwargs)

    adapter = get_runtime(backend)
    infer_llm = getattr(adapter, "infer_llm", None)
    if callable(infer_llm):
        return infer_llm(rh, input_or_prompt, **kwargs)
    return adapter.infer(rh, input_or_prompt, **kwargs)


def generate_LLM(rh: Any, prompt: Any, **overrides: Any) -> Any:
    backend = getattr(rh, "backend", None)
    if backend == "rbln":
        if _generate_rbln_llm is None:
            raise RuntimeError("RBLN LLM runtime is unavailable in this environment.")
        return _generate_rbln_llm(rh, prompt, **overrides)
    if backend == "tensorrt":
        if _generate_trt_llm is None:
            raise RuntimeError("TensorRT-LLM runtime is unavailable in this environment.")
        return _generate_trt_llm(rh, prompt, **overrides)
    if backend == "qb":
        raise NotImplementedError(
            "QB LLM runtime in main currently exposes low-level infer_LLM(...) only, not high-level text generation."
        )

    adapter = get_runtime(backend)
    return adapter.infer(rh, prompt, **overrides)


def destroy_runtime_LLM(rh: Any) -> None:
    backend = getattr(rh, "backend", None)
    if backend == "rbln":
        if _destroy_rbln_llm is None:
            raise RuntimeError("RBLN LLM runtime is unavailable in this environment.")
        return _destroy_rbln_llm(rh)
    if backend == "tensorrt":
        if _destroy_trt_llm is None:
            raise RuntimeError("TensorRT-LLM runtime is unavailable in this environment.")
        return _destroy_trt_llm(rh)

    adapter = get_runtime(backend)
    destroy_llm = getattr(adapter, "destroy_llm", None)
    if callable(destroy_llm):
        return destroy_llm(rh)
    return adapter.destroy(rh)

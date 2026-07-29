from __future__ import annotations
from typing import Any, Dict

from unified_sdk.build.registry import get_builder
from unified_sdk.types import BuildConfig, BuildResult, LLMBuildConfig

import warnings

try:
    from . import qb_build as _qb  # noqa: F401
except Exception as e:
    warnings.warn(f"QB backend disabled: {e!r}")
    _qb = None

try:
    from . import tensorrt_build as _tensorrt  # noqa: F401
except Exception as e:
    warnings.warn(f"TensorRT backend disabled: {e!r}")
    _tensorrt = None

try:
    from . import rbln_build as _rbln  # noqa: F401
except Exception as e:
    warnings.warn(f"RBLN backend disabled: {e!r}")
    _rbln = None

try:
    from . import warboy_build as _warboy  # noqa: F401
except Exception as e:
    warnings.warn(f"Warboy backend disabled: {e!r}")
    _warboy = None

try:
    from . import rngd_build as _rngd  # noqa: F401
except Exception as e:
    warnings.warn(f"RNGD backend disabled: {e!r}")
    _rngd = None

try:
    from .rbln_llm_build import build_llm as _build_rbln_llm
except Exception as e:
    warnings.warn(f"RBLN LLM build disabled: {e!r}")
    _build_rbln_llm = None

try:
    from . import rbln_llm_build as _rbln_llm
except Exception as e:
    warnings.warn(f"RBLN LLM build mapping disabled: {e!r}")
    _rbln_llm = None

try:
    from .tensorrt_llm_build import build_llm as _build_trt_llm
except Exception as e:
    warnings.warn(f"TensorRT-LLM build disabled: {e!r}")
    _build_trt_llm = None

try:
    from . import tensorrt_llm_build as _tensorrt_llm
except Exception as e:
    warnings.warn(f"TensorRT-LLM build mapping disabled: {e!r}")
    _tensorrt_llm = None


def _resolve_backend_name(target: Any) -> str:
    backend = getattr(target, "backend", target)
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError(
            "A backend name or config object with a non-empty .backend field is required. "
            "Examples: 'tensorrt', 'rbln', 'qb', 'warboy', 'rngd'."
        )
    return backend.strip()


def build_unified(cfg: BuildConfig) -> BuildResult:
    builder = get_builder(cfg.backend)
    return builder.build(cfg)


def build_unified_LLM(cfg: Any) -> BuildResult:
    backend = getattr(cfg, "backend", None)
    if backend == "rbln":
        if _build_rbln_llm is None:
            raise RuntimeError("RBLN LLM build backend is unavailable in this environment.")
        return _build_rbln_llm(cfg)  # type: ignore[arg-type]
    if backend == "tensorrt":
        if _build_trt_llm is None:
            raise RuntimeError("TensorRT-LLM build backend is unavailable in this environment.")
        return _build_trt_llm(cfg)  # type: ignore[arg-type]
    if backend == "rngd":
        # RNGD build surface is LLM semantic, but the adapter currently uses generic BuildConfig.
        return build_unified(cfg)  # type: ignore[arg-type]
    if backend == "qb":
        raise NotImplementedError(
            "QB LLM build is planned in main, but local LLM compile/fetch has not been normalized yet. "
            "Use the QB LLM runtime examples with a prepared .mxq artifact for now."
        )
    raise ValueError(f"Unsupported LLM build backend: {backend!r}")


def describe_build_api_mapping(target: Any) -> Dict[str, Any]:
    """Return vendor build API ==> Unified SDK build API mapping for a backend."""
    backend = _resolve_backend_name(target)
    if backend == "qb":
        if _qb is None:
            raise RuntimeError("QB build backend is unavailable in this environment.")
        return _qb.describe_api_mapping()
    if backend == "tensorrt":
        if _tensorrt is None:
            raise RuntimeError("TensorRT build backend is unavailable in this environment.")
        return _tensorrt.describe_api_mapping()
    if backend == "rbln":
        if _rbln is None:
            raise RuntimeError("RBLN build backend is unavailable in this environment.")
        return _rbln.describe_api_mapping()
    if backend == "warboy":
        if _warboy is None:
            raise RuntimeError("Warboy build backend is unavailable in this environment.")
        return _warboy.describe_api_mapping()
    if backend == "rngd":
        if _rngd is None:
            raise RuntimeError("RNGD build backend is unavailable in this environment.")
        return _rngd.describe_api_mapping()
    raise ValueError(f"Unsupported build mapping backend: {backend!r}")


def describe_build_api_mapping_LLM(target: Any) -> Dict[str, Any]:
    """Return vendor LLM build API ==> Unified SDK build API mapping for a backend."""
    backend = _resolve_backend_name(target)
    if backend == "rbln":
        if _rbln_llm is None:
            raise RuntimeError("RBLN LLM build backend is unavailable in this environment.")
        return _rbln_llm.describe_api_mapping()
    if backend == "tensorrt":
        if _tensorrt_llm is None:
            raise RuntimeError("TensorRT-LLM build backend is unavailable in this environment.")
        return _tensorrt_llm.describe_api_mapping()
    if backend == "rngd":
        if _rngd is None:
            raise RuntimeError("RNGD build backend is unavailable in this environment.")
        return _rngd.describe_api_mapping()
    if backend == "qb":
        raise NotImplementedError(
            "QB LLM build mapping in main is intentionally unsupported because QB LLM build remains unnormalized."
        )
    raise ValueError(f"Unsupported LLM build mapping backend: {backend!r}")

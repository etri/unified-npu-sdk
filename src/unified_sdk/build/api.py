from __future__ import annotations

from typing import Any, Dict

from unified_sdk.build.registry import get_builder, get_llm_builder
from unified_sdk.types import BuildConfig, BuildResult, LLMBuildConfig, LLMFetchConfig, LLMFetchResult

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
    from . import rbln_llm_build as _rbln_llm  # noqa: F401
except Exception as e:
    warnings.warn(f"RBLN LLM build mapping disabled: {e!r}")
    _rbln_llm = None

try:
    from . import tensorrt_llm_build as _tensorrt_llm  # noqa: F401
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


def fetch_unified_LLM(cfg: LLMFetchConfig) -> LLMFetchResult:
    builder = get_llm_builder(cfg.backend)
    return builder.fetch(cfg)


def build_unified_LLM(cfg: LLMBuildConfig) -> BuildResult:
    builder = get_llm_builder(cfg.backend)
    return builder.build(cfg)


def describe_build_api_mapping(target: Any) -> Dict[str, Any]:
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
    raise ValueError(f"Unsupported LLM build mapping backend: {backend!r}")

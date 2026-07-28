from __future__ import annotations
from typing import Any

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
    from .tensorrt_llm_build import build_llm as _build_trt_llm
except Exception as e:
    warnings.warn(f"TensorRT-LLM build disabled: {e!r}")
    _build_trt_llm = None


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

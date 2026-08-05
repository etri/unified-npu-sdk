"""
Unified SDK (Warboy-only worktree)
==================================

This worktree exposes a single FuriosaAI Warboy backend while preserving the
same layered SDK shape used by the unified project:

- ``frontends``: fetch / prepare helpers for provided ``.enf`` or quantized ONNX
- ``build``: compile or place the prepared Warboy artifact
- ``runtime``: create a runner and execute inference on ``.enf``

Quantization itself is intentionally treated as a separate prepare capability,
not as part of the build core.
"""

from .build import build_unified, describe_build_api_mapping
from .frontends import (
    WarboyFrontendBuildRequest,
    describe_frontend_api_mapping,
    list_model_zoo_targets,
    prepare_warboy_build_input,
    resolve_warboy_build_request,
)
from .options import WarboyBuildOptions, WarboyRuntimeOptions
from .runtime import create_runtime, describe_runtime_api_mapping, destroy_runtime, infer
from .types import BuildConfig, BuildResult, RuntimeConfig, RuntimeHandle

__all__ = [
    "BuildConfig",
    "BuildResult",
    "RuntimeConfig",
    "RuntimeHandle",
    "WarboyBuildOptions",
    "WarboyRuntimeOptions",
    "WarboyFrontendBuildRequest",
    "prepare_warboy_build_input",
    "resolve_warboy_build_request",
    "list_model_zoo_targets",
    "describe_frontend_api_mapping",
    "build_unified",
    "describe_build_api_mapping",
    "create_runtime",
    "infer",
    "destroy_runtime",
    "describe_runtime_api_mapping",
]

__version__ = "0.1.0"

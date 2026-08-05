"""
unified_sdk.frontends
---------------------
Frontend helpers for Warboy vision model preparation and fetch resolution.

In this worktree, quantization/preparation is treated as a capability that
precedes `build_unified(...)`, rather than being hidden inside the compiler
adapter itself.
"""

from .prepare_warboy_source import prepare_warboy_build_input
from .prepare_warboy_runtime_input import inspect_warboy_input_contract, prepare_warboy_runtime_input
from .resolve_warboy_build_request import describe_frontend_api_mapping, resolve_warboy_build_request
from .types import (
    PreparedWarboyBuildInput,
    PreparedWarboyCompileSource,
    PreparedWarboyRuntimeInput,
    ProvidedWarboyArtifact,
    ResolvedWarboyBuildRequest,
    WarboyFrontendBuildRequest,
)
from .warboy_model_zoo import fetch_model_zoo_enf, find_local_enf, list_model_zoo_targets, resolve_model_zoo_target

__all__ = [
    "PreparedWarboyBuildInput",
    "PreparedWarboyCompileSource",
    "PreparedWarboyRuntimeInput",
    "ProvidedWarboyArtifact",
    "ResolvedWarboyBuildRequest",
    "WarboyFrontendBuildRequest",
    "describe_frontend_api_mapping",
    "fetch_model_zoo_enf",
    "find_local_enf",
    "inspect_warboy_input_contract",
    "list_model_zoo_targets",
    "prepare_warboy_build_input",
    "prepare_warboy_runtime_input",
    "resolve_model_zoo_target",
    "resolve_warboy_build_request",
]

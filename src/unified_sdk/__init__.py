"""
Unified SDK (RNGD-only worktree)
================================

A unified SDK for compiling and serving LLMs on FuriosaAI RNGD NPUs.

Structure:
 - build:     Model preparation modules (fetch or fxb build)
 - runtime:   Runtime creation and text generation modules (furiosa_llm.LLM / explicit FXB)
 - backends:  Backend adapters (FuriosaAI RNGD only in this worktree)
 - frontends: Model import and conversion helpers

Preferred public surface:
 - `LLMBuildConfig`, `LLMRuntimeConfig`, `LLMRuntimeHandle`
 - `build_unified_LLM(cfg)`
 - `create_runtime_LLM(cfg)`, `generate_LLM(rh, prompt)`, `destroy_runtime_LLM(rh)`

Compatibility aliases such as `BuildConfig`, `RuntimeConfig`, `RuntimeHandle`,
and `infer_LLM(...)` remain available, but this worktree should be read as an
explicit RNGD LLM capability branch rather than a generic multi-capability one.
"""

__version__ = "0.1.0"

from unified_sdk.build import build_unified, build_unified_LLM
from unified_sdk.options import RNGDBuildOptions, RNGDRuntimeOptions
from unified_sdk.runtime import create_runtime_LLM, destroy_runtime_LLM, generate_LLM, infer_LLM
from unified_sdk.types import (
    BuildConfig,
    BuildResult,
    LLMBuildConfig,
    LLMRuntimeHandle,
    LLMRuntimeConfig,
    RuntimeConfig,
    RuntimeHandle,
)

__all__ = [
    "LLMBuildConfig",
    "LLMRuntimeConfig",
    "LLMRuntimeHandle",
    "BuildResult",
    "build_unified_LLM",
    "create_runtime_LLM",
    "generate_LLM",
    "destroy_runtime_LLM",
    "infer_LLM",
    "RNGDBuildOptions",
    "RNGDRuntimeOptions",
    "BuildConfig",
    "RuntimeConfig",
    "RuntimeHandle",
    "build_unified",
]

"""
Unified SDK (RNGD-only worktree)
================================

A unified SDK for compiling and serving LLMs on FuriosaAI RNGD NPUs.

Structure:
 - build:     Model preparation modules (fetch or fxb build)
 - runtime:   Runtime creation and text generation modules (furiosa_llm.LLM / explicit FXB)
 - backends:  Backend adapters (FuriosaAI RNGD only in this worktree)
 - frontends: Model import and conversion helpers

Note: RNGD is an LLM stack. `runtime.infer` performs text generation
(prompt -> text), unlike the numpy vision inference of the other worktrees.
"""

__version__ = "0.1.0"

from unified_sdk.options import RNGDBuildOptions, RNGDRuntimeOptions
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
    "BuildConfig",
    "BuildResult",
    "LLMBuildConfig",
    "LLMRuntimeHandle",
    "LLMRuntimeConfig",
    "RNGDBuildOptions",
    "RNGDRuntimeOptions",
    "RuntimeConfig",
    "RuntimeHandle",
]

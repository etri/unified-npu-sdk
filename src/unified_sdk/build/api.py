from __future__ import annotations
from typing import Any, Dict

from unified_sdk.build.registry import get_builder
from unified_sdk.types import BuildConfig, BuildResult, LLMBuildConfig

# Adapter auto-registration
from . import rngd_build as _rngd  # noqa: F401


def build_unified(cfg: BuildConfig) -> BuildResult:
    """Compatibility alias for the LLM-specific build surface.

    Prefer `build_unified_LLM(cfg)` in this RNGD-only worktree.
    """
    builder = get_builder(cfg.backend)
    return builder.build(cfg)


def build_unified_LLM(cfg: LLMBuildConfig | BuildConfig) -> BuildResult:
    """Primary LLM-specific build/fetch entrypoint for the RNGD-only backend."""
    builder = get_builder(cfg.backend)
    return builder.build(cfg)


build_unified_llm = build_unified_LLM


def describe_build_api_mapping() -> Dict[str, Any]:
    """Return vendor API ==> Unified SDK build API mapping for this backend."""
    return _rngd.describe_api_mapping()

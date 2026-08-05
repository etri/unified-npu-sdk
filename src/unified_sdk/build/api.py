from __future__ import annotations
from typing import Any, Dict

from unified_sdk.build.registry import get_builder, get_llm_builder
from unified_sdk.types import BuildConfig, BuildResult, LLMBuildConfig

# Adapter auto-registration
from . import rbln_build as _rbln  # noqa: F401
from . import rbln_llm_build as _rbln_llm  # noqa: F401


def build_unified(cfg: BuildConfig) -> BuildResult:
    builder = get_builder(cfg.backend)
    return builder.build(cfg)


def build_unified_LLM(cfg: LLMBuildConfig) -> BuildResult:
    builder = get_llm_builder(cfg.backend)
    return builder.build(cfg)


def describe_build_api_mapping() -> Dict[str, Any]:
    """Return vendor API ==> Unified SDK build API mapping for this backend."""
    return _rbln.describe_api_mapping()


def describe_build_api_mapping_LLM() -> Dict[str, Any]:
    """Return vendor LLM build API ==> Unified SDK build API mapping for this backend."""
    return _rbln_llm.describe_api_mapping()

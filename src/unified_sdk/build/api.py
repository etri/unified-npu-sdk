from __future__ import annotations
from typing import Any, Dict

from unified_sdk.build.registry import get_builder
from unified_sdk.types import BuildConfig, BuildResult

# Adapter auto-registration
from . import tensorrt_build as _tensorrt  # noqa: F401


def build_unified(cfg: BuildConfig) -> BuildResult:
    builder = get_builder(cfg.backend)
    return builder.build(cfg)


def describe_build_api_mapping() -> Dict[str, Any]:
    """Return vendor API ==> Unified SDK build API mapping for this backend."""
    return _tensorrt.describe_api_mapping()

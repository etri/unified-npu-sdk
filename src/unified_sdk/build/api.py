from __future__ import annotations
from unified_sdk.build.registry import get_builder
from unified_sdk.types import BuildConfig, BuildResult

# Adapter auto-registration
from . import rngd_build as _rngd  # noqa: F401


def build_unified(cfg: BuildConfig) -> BuildResult:
    builder = get_builder(cfg.backend)
    return builder.build(cfg)

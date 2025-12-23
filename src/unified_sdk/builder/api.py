from __future__ import annotations
from unified_sdk.build.registry import get_builder
from unified_sdk.types import BuildConfig, BuildResult

# 어댑터 자동 등록 (TensorRT 등)
from . import tensorrt_build as _tensorrt  # noqa: F401

def build_unified(cfg: BuildConfig) -> BuildResult:
    builder = get_builder(cfg.backend)
    return builder.build(cfg)


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from unified_sdk.options import RNGDBuildOptions, RNGDRuntimeOptions

BuildBackendName = Literal["rngd"]
RuntimeBackendName = Literal["rngd"]


@dataclass(kw_only=True)
class CoreBuildConfig:
    """Backend-agnostic build inputs shared by the RNGD LLM capability."""

    model_or_path: Any
    out_dir: str | Path = "artifacts"
    model_name: str = "model"


@dataclass(kw_only=True)
class LLMBuildConfig(CoreBuildConfig):
    """Explicit LLM capability config for the RNGD-only worktree."""

    backend: BuildBackendName = "rngd"
    backend_options: RNGDBuildOptions | None = None


@dataclass(kw_only=True)
class BuildConfig(LLMBuildConfig):
    """Backward-compatible alias for the RNGD build surface."""


@dataclass
class BuildResult:
    backend: str
    compiled_model_path: str
    meta_data: Dict[str, Any]


@dataclass(kw_only=True)
class CoreRuntimeConfig:
    """Backend-agnostic generation defaults for LLM runtimes."""

    engine_path: str | Path
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    min_tokens: int = 0


@dataclass(kw_only=True)
class LLMRuntimeConfig(CoreRuntimeConfig):
    """Explicit LLM runtime config for the RNGD-only worktree."""

    backend: RuntimeBackendName = "rngd"
    backend_options: RNGDRuntimeOptions | None = None


@dataclass(kw_only=True)
class RuntimeConfig(LLMRuntimeConfig):
    """Backward-compatible alias for the RNGD runtime surface."""


@dataclass(kw_only=True)
class CoreRuntimeHandle:
    backend: str
    engine_path: str
    ctx: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class LLMRuntimeHandle(CoreRuntimeHandle):
    """LLM-specific runtime handle for text-generation backends."""


@dataclass(kw_only=True)
class RuntimeHandle(LLMRuntimeHandle):
    """Backward-compatible alias for runtime handles."""

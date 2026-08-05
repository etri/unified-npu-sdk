from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

from unified_sdk.options import WarboyBuildOptions, WarboyRuntimeOptions

BuildBackendName = Literal["warboy"]
RuntimeBackendName = Literal["warboy"]


@dataclass(kw_only=True)
class CoreBuildConfig:
    """Backend-agnostic build inputs for Warboy vision compilation."""

    model_or_path: Any
    out_dir: str | Path = "builds"
    model_name: str = "model"
    input_name: str = "input"
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)


@dataclass(kw_only=True)
class BuildConfig(CoreBuildConfig):
    backend: BuildBackendName = "warboy"
    backend_options: WarboyBuildOptions | None = None


@dataclass
class BuildResult:
    backend: str
    compiled_model_path: str
    meta_data: Dict[str, Any]


@dataclass(kw_only=True)
class CoreRuntimeConfig:
    engine_path: str | Path
    input_name: str
    output_name: str
    input_shape: Tuple[int, ...]


@dataclass(kw_only=True)
class RuntimeConfig(CoreRuntimeConfig):
    backend: RuntimeBackendName = "warboy"
    backend_options: WarboyRuntimeOptions | None = None


@dataclass(kw_only=True)
class CoreRuntimeHandle:
    backend: str
    engine_path: str
    input_name: str
    output_name: str
    input_shape: Tuple[int, ...]
    ctx: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class RuntimeHandle(CoreRuntimeHandle):
    pass

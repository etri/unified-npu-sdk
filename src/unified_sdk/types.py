from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Tuple, TypeAlias

import numpy as np

from unified_sdk.options import WarboyBuildOptions, WarboyRuntimeOptions

if TYPE_CHECKING:
    from unified_sdk.frontends.types import PreparedWarboyBuildInput

BuildBackendName = Literal["warboy"]
RuntimeBackendName = Literal["warboy"]
InferOutput: TypeAlias = np.ndarray | list[np.ndarray]


@dataclass(kw_only=True)
class CoreBuildConfig:
    """Backend-agnostic build inputs for Warboy vision compilation."""

    model_or_path: str | Path
    out_dir: str | Path = "builds"
    model_name: str = "model"
    input_name: str = "input"
    input_shape: Tuple[int, ...] | None = (1, 3, 224, 224)


@dataclass(kw_only=True)
class BuildConfig(CoreBuildConfig):
    backend: BuildBackendName = "warboy"
    backend_options: WarboyBuildOptions | None = None
    prepared_input: "PreparedWarboyBuildInput | None" = None


@dataclass
class BuildResult:
    backend: str
    compiled_model_path: str
    meta_data: Dict[str, Any]


@dataclass(kw_only=True)
class CoreRuntimeConfig:
    engine_path: str | Path
    input_name: str
    output_name: str | None = None
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
    output_name: str | None
    input_shape: Tuple[int, ...]
    ctx: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class RuntimeHandle(CoreRuntimeHandle):
    pass

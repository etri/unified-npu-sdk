from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple

from unified_sdk.options import (
    RBLNLLMBuildOptions,
    RBLNLLMRuntimeOptions,
    RBLNVisionBuildOptions,
    RBLNVisionRuntimeOptions,
)

if TYPE_CHECKING:
    from unified_sdk.frontends.types import PreparedRBLNVisionBuildInput

BuildBackendName = Literal["rbln"]
RuntimeBackendName = Literal["rbln"]


@dataclass(kw_only=True)
class CoreBuildConfig:
    model_or_path: Any
    out_dir: str | Path = "build"
    model_name: str = "model"
    input_name: str = "input"
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)


@dataclass(kw_only=True)
class BuildConfig(CoreBuildConfig):
    backend: BuildBackendName = "rbln"
    backend_options: RBLNVisionBuildOptions | None = None
    prepared_input: "PreparedRBLNVisionBuildInput | None" = None
    bucketing_shapes: Optional[list[Tuple[int, ...]]] = None
    extra: Optional[Dict[str, Any]] = None  # legacy compatibility fallback


@dataclass
class BuildResult:
    backend: str
    compiled_model_path: str
    meta_data: Dict[str, Any]


@dataclass(kw_only=True)
class CoreRuntimeConfig:
    engine_path: str | Path
    input_name: str
    output_name: str = "output"
    input_shape: Tuple[int, ...]


@dataclass(kw_only=True)
class RuntimeConfig(CoreRuntimeConfig):
    backend: RuntimeBackendName = "rbln"
    backend_options: RBLNVisionRuntimeOptions | None = None
    extra: Optional[Dict[str, Any]] = None  # legacy compatibility fallback


@dataclass(kw_only=True)
class RuntimeHandle:
    backend: str
    engine_path: str
    input_name: str
    output_name: str
    input_shape: Tuple[int, ...]
    ctx: Dict[str, Any] = field(default_factory=dict)


LLMBuildBackendName = Literal["rbln"]
LLMRuntimeBackendName = Literal["rbln"]


@dataclass(kw_only=True)
class CoreLLMBuildConfig:
    model_or_path: str | Path
    out_dir: str | Path = "artifacts"
    model_name: str = "model"
    batch_size: int = 1
    max_model_len: int = 512
    num_devices: int = 1


@dataclass(kw_only=True)
class LLMBuildConfig(CoreLLMBuildConfig):
    backend: LLMBuildBackendName = "rbln"
    backend_options: RBLNLLMBuildOptions | None = None
    extra: Optional[Dict[str, Any]] = None  # legacy compatibility fallback


@dataclass(kw_only=True)
class CoreLLMRuntimeConfig:
    engine_path: str | Path
    tokenizer_path: Optional[str | Path] = None
    tensor_parallel_size: int = 1
    max_model_len: int = 512
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    min_tokens: int = 0


@dataclass(kw_only=True)
class LLMRuntimeConfig(CoreLLMRuntimeConfig):
    backend: LLMRuntimeBackendName = "rbln"
    backend_options: RBLNLLMRuntimeOptions | None = None
    extra: Optional[Dict[str, Any]] = None  # legacy compatibility fallback


@dataclass(kw_only=True)
class LLMRuntimeHandle:
    backend: str
    engine_path: str
    ctx: Dict[str, Any] = field(default_factory=dict)

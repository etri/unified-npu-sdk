from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Tuple

from unified_sdk.options import (
    TensorRTLLMBuildOptions,
    TensorRTLLMRuntimeOptions,
    TensorRTVisionBuildOptions,
    TensorRTVisionRuntimeOptions,
)

if TYPE_CHECKING:
    from unified_sdk.frontends.types import PreparedTensorRTLLMBuildInput, PreparedTensorRTVisionBuildInput


BuildBackendName = Literal["tensorrt"]
RuntimeBackendName = Literal["tensorrt"]


@dataclass(kw_only=True)
class CoreBuildConfig:
    model_or_path: Any
    out_dir: str | Path = "build_output"
    model_name: str = "model"


@dataclass(kw_only=True)
class BuildConfig(CoreBuildConfig):
    backend: BuildBackendName = "tensorrt"
    backend_options: TensorRTVisionBuildOptions | None = None
    prepared_input: "PreparedTensorRTVisionBuildInput | None" = None


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
    backend: RuntimeBackendName = "tensorrt"
    backend_options: TensorRTVisionRuntimeOptions | None = None


@dataclass(kw_only=True)
class RuntimeHandle:
    backend: str
    engine_path: str
    input_name: str
    output_name: str
    input_shape: Tuple[int, ...]
    ctx: Dict[str, Any] = field(default_factory=dict)


LLMBuildBackendName = Literal["tensorrt"]
LLMRuntimeBackendName = Literal["tensorrt"]


@dataclass(kw_only=True)
class CoreLLMBuildConfig:
    model_or_path: str | Path
    out_dir: str | Path = "artifacts"
    model_name: str = "model"


@dataclass(kw_only=True)
class LLMBuildConfig(CoreLLMBuildConfig):
    backend: LLMBuildBackendName = "tensorrt"
    backend_options: TensorRTLLMBuildOptions | None = None
    prepared_input: "PreparedTensorRTLLMBuildInput | None" = None


@dataclass(kw_only=True)
class CoreLLMRuntimeConfig:
    engine_path: str | Path
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 50
    min_tokens: int = 0


@dataclass(kw_only=True)
class LLMRuntimeConfig(CoreLLMRuntimeConfig):
    backend: LLMRuntimeBackendName = "tensorrt"
    backend_options: TensorRTLLMRuntimeOptions | None = None


@dataclass(kw_only=True)
class LLMRuntimeHandle:
    backend: str
    engine_path: str
    ctx: Dict[str, Any] = field(default_factory=dict)

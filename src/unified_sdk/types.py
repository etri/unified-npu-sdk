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


@dataclass(init=False, kw_only=True)
class CoreLLMRuntimeConfig:
    model_ref_or_path: str | Path
    max_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 50
    min_tokens: int = 0

    def __init__(
        self,
        *,
        model_ref_or_path: str | Path | None = None,
        engine_path: str | Path | None = None,
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 50,
        min_tokens: int = 0,
    ) -> None:
        if model_ref_or_path is None and engine_path is None:
            raise ValueError("LLMRuntimeConfig requires model_ref_or_path (preferred) or legacy engine_path")
        if model_ref_or_path is not None and engine_path is not None and str(model_ref_or_path) != str(engine_path):
            raise ValueError("model_ref_or_path and legacy engine_path must match when both are provided")
        self.model_ref_or_path = model_ref_or_path if model_ref_or_path is not None else engine_path  # type: ignore[assignment]
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_tokens = min_tokens

    @property
    def engine_path(self) -> str | Path:
        return self.model_ref_or_path


@dataclass(init=False, kw_only=True)
class LLMRuntimeConfig(CoreLLMRuntimeConfig):
    backend: LLMRuntimeBackendName = "tensorrt"
    backend_options: TensorRTLLMRuntimeOptions | None = None

    def __init__(
        self,
        *,
        backend: LLMRuntimeBackendName = "tensorrt",
        backend_options: TensorRTLLMRuntimeOptions | None = None,
        model_ref_or_path: str | Path | None = None,
        engine_path: str | Path | None = None,
        max_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 50,
        min_tokens: int = 0,
    ) -> None:
        super().__init__(
            model_ref_or_path=model_ref_or_path,
            engine_path=engine_path,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_tokens=min_tokens,
        )
        self.backend = backend
        self.backend_options = backend_options


@dataclass(kw_only=True)
class LLMRuntimeHandle:
    backend: str
    engine_path: str
    ctx: Dict[str, Any] = field(default_factory=dict)

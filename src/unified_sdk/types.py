from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Tuple


BuildBackendName = Literal["tensorrt", "rbln", "warboy", "rngd", "qb"]
RuntimeBackendName = Literal["tensorrt", "rbln", "warboy", "rngd", "qb"]

Precision = Literal["fp32", "fp16", "int8"]


@dataclass
class BuildConfig:
    backend: BuildBackendName
    model_or_path: Any
    out_dir: str | Path = "build"
    model_name: str = "model"
    precision: Precision = "fp16"
    input_name: str = "input.1"
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    bucketing_shapes: list[Tuple[int, ...]] | None = None
    min_input_shape: Tuple[int, ...] = (1, 3, 256, 192)
    opt_input_shape: Tuple[int, ...] = (4, 3, 256, 192)
    max_input_shape: Tuple[int, ...] = (30, 3, 256, 192)
    calib_data_path: str | Path | None = None
    use_random_calib: bool = False
    batch_size: int = 1
    num_devices: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: int | None = None
    backend_options: Any = None
    prepared_input: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildResult:
    backend: str
    compiled_model_path: str
    meta_data: Dict[str, Any]


@dataclass
class RuntimeConfig:
    backend: RuntimeBackendName
    engine_path: str | Path
    input_name: str = "input"
    output_name: str = "output"
    input_shape: Tuple[int, ...] = (1,)
    use_execute_v3: bool = True
    fxb_path: str | Path | None = None
    devices: Any = None
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    min_tokens: int = 0
    backend_options: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeHandle:
    backend: str
    engine_path: str
    input_name: str = "input"
    output_name: str = "output"
    input_shape: Tuple[int, ...] = (1,)
    ctx: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchParam:
    sequence_length: int
    cache_size: int = 0
    cache_id: int = 0


@dataclass
class LLMFetchConfig:
    backend: BuildBackendName
    model_ref: str | Path
    prepared_input: Any = None


@dataclass
class LLMFetchResult:
    backend: str
    model_ref_or_path: str
    meta_data: Dict[str, Any]


@dataclass
class LLMBuildConfig:
    backend: BuildBackendName
    model_or_path: Any
    out_dir: str | Path = "artifacts"
    model_name: str = "model"
    batch_size: int = 1
    num_devices: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: int = 512
    backend_options: Any = None
    prepared_input: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(init=False)
class LLMRuntimeConfig:
    backend: RuntimeBackendName
    model_ref_or_path: str | Path
    tokenizer_path: str | Path | None
    max_model_len: int
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    min_tokens: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    batch_size: int
    fxb_path: str | Path | None
    devices: Any
    backend_options: Any
    prepared_fetch_input: Any
    extra: Dict[str, Any]

    def __init__(
        self,
        *,
        backend: RuntimeBackendName,
        model_ref_or_path: str | Path | None = None,
        engine_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
        max_model_len: int = 512,
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        min_tokens: int = 0,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        batch_size: int = 1,
        fxb_path: str | Path | None = None,
        devices: Any = None,
        backend_options: Any = None,
        prepared_fetch_input: Any = None,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        if model_ref_or_path is None and engine_path is None:
            raise ValueError("LLMRuntimeConfig requires model_ref_or_path (preferred) or legacy engine_path")
        if model_ref_or_path is not None and engine_path is not None and str(model_ref_or_path) != str(engine_path):
            raise ValueError("model_ref_or_path and legacy engine_path must match when both are provided")

        self.backend = backend
        self.model_ref_or_path = model_ref_or_path if model_ref_or_path is not None else engine_path  # type: ignore[assignment]
        self.tokenizer_path = tokenizer_path
        self.max_model_len = max_model_len
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_tokens = min_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.batch_size = batch_size
        self.fxb_path = fxb_path
        self.devices = devices
        self.backend_options = backend_options
        self.prepared_fetch_input = prepared_fetch_input
        self.extra = dict(extra or {})

    @property
    def engine_path(self) -> str | Path:
        return self.model_ref_or_path


@dataclass(init=False)
class LLMRuntimeHandle:
    backend: str
    model_ref_or_path: str
    tokenizer_path: str | None
    ctx: Dict[str, Any]

    def __init__(
        self,
        *,
        backend: str,
        model_ref_or_path: str | None = None,
        engine_path: str | None = None,
        tokenizer_path: str | None = None,
        ctx: Dict[str, Any] | None = None,
    ) -> None:
        if model_ref_or_path is None and engine_path is None:
            raise ValueError("LLMRuntimeHandle requires model_ref_or_path (preferred) or legacy engine_path")
        if model_ref_or_path is not None and engine_path is not None and model_ref_or_path != engine_path:
            raise ValueError("model_ref_or_path and legacy engine_path must match when both are provided")
        self.backend = backend
        self.model_ref_or_path = model_ref_or_path if model_ref_or_path is not None else engine_path  # type: ignore[assignment]
        self.tokenizer_path = tokenizer_path
        self.ctx = dict(ctx or {})

    @property
    def engine_path(self) -> str:
        return self.model_ref_or_path


CoreRuntimeConfig = RuntimeConfig
CoreRuntimeHandle = RuntimeHandle

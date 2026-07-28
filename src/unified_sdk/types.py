from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

# main 통합 단계에서는 vendor별 구현 편차가 크므로, 공통 타입을 넓게 유지합니다.
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
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMRuntimeConfig:
    backend: RuntimeBackendName
    engine_path: str | Path
    tokenizer_path: str | Path | None = None
    max_model_len: int = 512
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    min_tokens: int = 0
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    batch_size: int = 1
    fxb_path: str | Path | None = None
    devices: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMRuntimeHandle:
    backend: str
    engine_path: str
    tokenizer_path: str | None = None
    ctx: Dict[str, Any] = field(default_factory=dict)

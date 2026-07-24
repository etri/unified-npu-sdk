from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

BuildBackendName = Literal["rbln"]
RuntimeBackendName = Literal["rbln"]

Precision = Literal["fp32", "fp16"]

@dataclass
class BuildConfig:
    backend: BuildBackendName
    model_or_path: Any                      # torch.nn.Module 권장
    out_dir: str | Path = "build"
    model_name: str = "model"
    precision: Precision = "fp16"
    input_name: str = "input"
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    # RBLN bucketing 옵션: 여러 입력 shape를 동시에 컴파일
    bucketing_shapes: Optional[List[Tuple[int, ...]]] = None
    extra: Optional[Dict[str, Any]] = None

@dataclass
class BuildResult:
    backend: str
    compiled_model_path: str
    meta_data: Dict[str, Any]

@dataclass
class RuntimeConfig:
    backend: RuntimeBackendName
    engine_path: str | Path
    input_name: str
    output_name: str
    input_shape: Tuple[int, ...]
    extra: Optional[Dict[str, Any]] = None  # device, tensor_type, activate_profiler, timeout, allow_dynamic_shape

@dataclass
class RuntimeHandle:
    backend: str
    engine_path: str
    input_name: str
    output_name: str
    input_shape: Tuple[int, ...]
    ctx: Dict[str, Any] = field(default_factory=dict)


# RBLN LLM track

LLMBuildBackendName = Literal["rbln"]
LLMRuntimeBackendName = Literal["rbln"]


@dataclass
class LLMBuildConfig:
    backend: LLMBuildBackendName
    model_or_path: str | Path
    out_dir: str | Path = "artifacts"
    model_name: str = "model"
    batch_size: int = 1
    max_model_len: int = 512
    num_devices: int = 1
    extra: Optional[Dict[str, Any]] = None  # build_mode(fetch|optimum_compile), trust_remote_code, revision 등


@dataclass
class LLMRuntimeConfig:
    backend: LLMRuntimeBackendName
    engine_path: str | Path                    # model id, local HF path, or precompiled RBLN dir
    tokenizer_path: Optional[str | Path] = None
    tensor_parallel_size: int = 1
    max_model_len: int = 512
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    min_tokens: int = 0
    extra: Optional[Dict[str, Any]] = None     # runtime_impl(vllm|optimum), block_size, trust_remote_code 등


@dataclass
class LLMRuntimeHandle:
    backend: str
    engine_path: str
    ctx: Dict[str, Any] = field(default_factory=dict)

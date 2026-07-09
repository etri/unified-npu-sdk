from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

BuildBackendName = Literal["rngd"]
RuntimeBackendName = Literal["rngd"]

# RNGD 는 LLM 스택이라 vision 백엔드(rbln/qb/warboy)의 input_shape/precision 대신
# LLM 준비/서빙에 맞는 필드를 노출한다.

@dataclass
class BuildConfig:
    backend: BuildBackendName
    # HuggingFace 모델 id (예: 'furiosa-ai/Qwen2.5-0.5B-Instruct') 또는 로컬 모델/아티팩트 경로
    model_or_path: Any
    out_dir: str | Path = "artifacts"
    model_name: str = "model"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None  # compile(bool), bucket_config 등 ArtifactBuilder 옵션

@dataclass
class BuildResult:
    backend: str
    compiled_model_path: str            # 아티팩트 디렉터리 경로 또는 HF 모델 id
    meta_data: Dict[str, Any]

@dataclass
class RuntimeConfig:
    backend: RuntimeBackendName
    engine_path: str | Path             # 아티팩트 dir 또는 HF 모델 id
    devices: Optional[str] = None       # 예: 'npu:0'. 미지정 시 furiosa-llm 기본 선택
    # 기본 SamplingParams (호출별로 override 가능)
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    min_tokens: int = 0
    extra: Optional[Dict[str, Any]] = None

@dataclass
class RuntimeHandle:
    backend: str
    engine_path: str
    ctx: Dict[str, Any] = field(default_factory=dict)

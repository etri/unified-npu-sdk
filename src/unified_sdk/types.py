from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

# 빌드/런타임 이름을 분리해 둡니다 (필요 시 추가)
BuildBackendName = Literal["tensorrt", "rbln"]      # "rebellions", "furiosa", "onnxrt", ...
RuntimeBackendName = Literal["tensorrt", "rbln"]    # 이미 사용 중

Precision = Literal["fp32", "fp16", "int8"]  # int8은 추후 calibration 확장

@dataclass
class BuildConfig:
    backend: BuildBackendName
    # 251212 jsk : torch.nn.module 오브젝트 받기 위함. 
    # model_or_path: str | Path
    model_or_path: Any
    out_dir: str | Path = "build"
    model_name: str = "model"
    precision: Precision = "fp16"
    # TensorRT 전용(다른 백엔드는 무시/자체 해석):
    input_name: str = "input.1"
    min_input_shape: Tuple[int, ...] = (1, 3, 256, 192)
    opt_input_shape: Tuple[int, ...] = (4, 3, 256, 192)
    max_input_shape: Tuple[int, ...] = (30, 3, 256, 192)
    extra: Dict[str, Any] = None

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
    use_execute_v3: bool = True  # TensorRT에만 의미 있음(다른 백엔드는 무시)
    extra: Dict[str, Any] = None

@dataclass
class RuntimeHandle:
    backend: str
    engine_path: str
    input_name: str
    output_name: str
    input_shape: Tuple[int, ...]
    # 아래는 백엔드 전용 객체/버퍼를 담는 캐리어
    ctx: Dict[str, Any]

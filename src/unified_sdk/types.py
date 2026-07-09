from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

BuildBackendName = Literal["qb"]
RuntimeBackendName = Literal["qb"]

# Mobilint ARISE(QB) 실행 모델(.mxq)은 qubee 컴파일러가 int8로 양자화한다.
Precision = Literal["int8"]

@dataclass
class BuildConfig:
    backend: BuildBackendName
    model_or_path: Any                      # ONNX 경로 / torch.nn.Module / 기존 .mxq 경로
    out_dir: str | Path = "builds"
    model_name: str = "model"
    precision: Precision = "int8"
    input_name: str = "input"
    input_shape: Tuple[int, ...] = (1, 3, 224, 224)
    # qubee 양자화 컴파일용 calibration 데이터셋 메타 파일(.txt/.json).
    # 미지정 시 build 어댑터가 random calibration(use_random_calib)로 대체.
    calib_data_path: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None  # quantize_method, core_mode, use_random_calib, save_sample 등

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
    extra: Optional[Dict[str, Any]] = None  # device, core_mode, dev, allow_dynamic_shape 등

@dataclass
class RuntimeHandle:
    backend: str
    engine_path: str
    input_name: str
    output_name: str
    input_shape: Tuple[int, ...]
    ctx: Dict[str, Any] = field(default_factory=dict)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PreparedTensorRTVisionSourceKind = Literal["provided_artifact", "compile_source"]
ResolvedTensorRTVisionBuildKind = Literal["provided_artifact", "onnx_path", "torchvision_export", "pth_export"]
PreparedTensorRTLLMFetchKind = Literal["runtime_model_ref"]
ResolvedTensorRTLLMFetchKind = Literal["runtime_model_ref"]
PreparedTensorRTLLMBuildKind = Literal["artifact_build"]
ResolvedTensorRTLLMBuildKind = Literal["artifact_build"]
PreparedTensorRTLLMSourceKind = Literal["model_id", "local_model_path", "local_artifact_dir", "local_checkpoint_dir"]
PreparedTensorRTLLMCompileVariant = Literal["model_ref_api", "checkpoint_dir_cli"]


@dataclass(frozen=True)
class TensorRTVisionFrontendBuildRequest:
    model_name: str
    models_dir: Path
    out_dir: Path
    precision: Literal["fp32", "fp16", "int8"] = "fp32"
    provided_engine: Path | None = None
    onnx_path: Path | None = None
    weights_path: Path | None = None
    export_onnx_path: Path | None = None
    model_zoo_model: str | None = None
    pretrained: bool = False
    require_onnx: bool = False
    input_name: str = "input"
    min_input_shape: tuple[int, ...] = (1, 3, 224, 224)
    opt_input_shape: tuple[int, ...] = (1, 3, 224, 224)
    max_input_shape: tuple[int, ...] = (1, 3, 224, 224)


@dataclass(frozen=True)
class ProvidedTensorRTArtifact:
    source_path: Path
    destination_path: Path


@dataclass(frozen=True)
class PreparedTensorRTCompileSource:
    source_path: Path
    source_label: str
    provenance_kind: ResolvedTensorRTVisionBuildKind
    provenance_detail: str
    input_name: str
    min_input_shape: tuple[int, ...]
    opt_input_shape: tuple[int, ...]
    max_input_shape: tuple[int, ...]


@dataclass(frozen=True)
class PreparedTensorRTVisionBuildInput:
    kind: PreparedTensorRTVisionSourceKind
    provided_artifact: ProvidedTensorRTArtifact | None = None
    compile_source: PreparedTensorRTCompileSource | None = None


@dataclass(frozen=True)
class ResolvedTensorRTVisionBuildRequest:
    model_or_path: str
    source_description: str
    kind: ResolvedTensorRTVisionBuildKind
    prepared_input: PreparedTensorRTVisionBuildInput


@dataclass(frozen=True)
class TensorRTLLMFrontendFetchRequest:
    model_ref: str | Path


@dataclass(frozen=True)
class PreparedTensorRTLLMFetchInput:
    kind: PreparedTensorRTLLMFetchKind
    model_ref: str
    source_kind: PreparedTensorRTLLMSourceKind = "model_id"
    source_path: Path | None = None


@dataclass(frozen=True)
class ResolvedTensorRTLLMFetchRequest:
    source_description: str
    kind: ResolvedTensorRTLLMFetchKind
    prepared_input: PreparedTensorRTLLMFetchInput


@dataclass(frozen=True)
class TensorRTLLMFrontendBuildRequest:
    model_ref: str | Path
    out_dir: Path
    model_name: str


@dataclass(frozen=True)
class PreparedTensorRTLLMBuildInput:
    kind: PreparedTensorRTLLMBuildKind
    model_ref: str
    source_kind: PreparedTensorRTLLMSourceKind = "model_id"
    source_path: Path | None = None
    artifact_dir: Path | None = None
    compile_variant: PreparedTensorRTLLMCompileVariant | None = None
    checkpoint_dir: Path | None = None


@dataclass(frozen=True)
class ResolvedTensorRTLLMBuildRequest:
    source_description: str
    kind: ResolvedTensorRTLLMBuildKind
    prepared_input: PreparedTensorRTLLMBuildInput

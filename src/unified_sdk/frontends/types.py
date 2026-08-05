from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


PreparedWarboySourceKind = Literal["provided_artifact", "compile_source"]
ResolvedWarboyBuildKind = Literal["provided_artifact", "local_enf", "model_zoo_enf", "quantized_onnx"]


@dataclass(frozen=True)
class WarboyFrontendBuildRequest:
    model_name: str
    models_dir: Path
    target_npu: str
    provided_enf: Path | None = None
    from_onnx: Path | None = None
    require_enf: bool = False


@dataclass(frozen=True)
class ProvidedWarboyArtifact:
    source_path: Path
    destination_path: Path


@dataclass(frozen=True)
class PreparedWarboyCompileSource:
    source: Any
    source_label: str


@dataclass(frozen=True)
class PreparedWarboyBuildInput:
    kind: PreparedWarboySourceKind
    provided_artifact: ProvidedWarboyArtifact | None = None
    compile_source: PreparedWarboyCompileSource | None = None


@dataclass(frozen=True)
class ResolvedWarboyBuildRequest:
    model_or_path: str
    source_description: str
    kind: ResolvedWarboyBuildKind
    prepared_input: PreparedWarboyBuildInput

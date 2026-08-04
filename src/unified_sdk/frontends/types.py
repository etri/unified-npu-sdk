from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


PreparedQBSourceKind = Literal["provided_artifact", "compile_source"]
ResolvedQBBuildKind = Literal["model_zoo_fetch", "provided_artifact", "local_onnx", "weights_export"]


@dataclass(frozen=True)
class QBFrontendBuildRequest:
    model_name: str
    models_dir: Path
    product: str
    core_mode: str
    from_pth: Path | None = None
    from_onnx: Path | None = None
    provided_mxq: Path | None = None
    export_onnx_path: Path | None = None
    input_name: str = "input"
    input_shape: tuple[int, ...] = (1, 3, 224, 224)
    require_mxq: bool = False


@dataclass(frozen=True)
class ProvidedQBArtifact:
    source_path: Path
    destination_path: Path


@dataclass(frozen=True)
class PreparedQBCompileSource:
    source: Any
    source_label: str


@dataclass(frozen=True)
class PreparedQBBuildInput:
    kind: PreparedQBSourceKind
    provided_artifact: ProvidedQBArtifact | None = None
    compile_source: PreparedQBCompileSource | None = None


@dataclass(frozen=True)
class ResolvedQBBuildRequest:
    model_or_path: str
    source_description: str
    kind: ResolvedQBBuildKind

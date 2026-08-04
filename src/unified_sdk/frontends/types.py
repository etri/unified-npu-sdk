from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


PreparedQBSourceKind = Literal["provided_artifact", "compile_source"]


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

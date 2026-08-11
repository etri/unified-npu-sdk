from __future__ import annotations

import shutil
from pathlib import Path

from unified_sdk.frontends.types import ProvidedQBArtifact


def place_provided_qb_artifact(artifact: ProvidedQBArtifact) -> Path:
    src = artifact.source_path
    dst = artifact.destination_path
    if not src.is_file():
        raise FileNotFoundError(f"Provided .mxq not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    return dst

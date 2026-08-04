from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from unified_sdk.options import QBSequenceRuntimeOptions

SequenceRuntimeBackendName = Literal["qb"]


@dataclass
class SequenceRuntimeConfig:
    backend: SequenceRuntimeBackendName
    engine_path: str | Path
    input_name: str
    output_name: str
    input_shape: Tuple[int, ...]
    backend_options: QBSequenceRuntimeOptions | None = None
    extra: Optional[Dict[str, Any]] = None  # deprecated compatibility fallback; new code should use backend_options


@dataclass(frozen=True)
class SequenceBatchParam:
    sequence_length: int
    cache_size: int = 0
    cache_id: int = 0


@dataclass
class SequenceRuntimeHandle:
    backend: str
    engine_path: str
    input_name: str
    output_name: str
    input_shape: Tuple[int, ...]
    ctx: Dict[str, Any] = field(default_factory=dict)

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from unified_sdk.options import QBSequenceRuntimeOptions
from unified_sdk.types import CoreRuntimeConfig, CoreRuntimeHandle

SequenceRuntimeBackendName = Literal["qb"]


@dataclass(kw_only=True)
class SequenceRuntimeConfig(CoreRuntimeConfig):
    backend: SequenceRuntimeBackendName = "qb"
    backend_options: QBSequenceRuntimeOptions | None = None


@dataclass(frozen=True)
class SequenceBatchParam:
    sequence_length: int
    cache_size: int = 0
    cache_id: int = 0


@dataclass(kw_only=True)
class SequenceRuntimeHandle(CoreRuntimeHandle):
    pass

from __future__ import annotations

from typing import Any, Optional, Sequence

from unified_sdk.runtime._qb_common import parse_non_negative_int
from unified_sdk.sequence_runtime.types import SequenceBatchParam


def normalize_batch_params(
    batch_params: Optional[Sequence[Any]],
    qbruntime_module: Any,
) -> Optional[list[Any]]:
    if batch_params is None:
        return None

    batch_param_cls = getattr(qbruntime_module, "BatchParam", None)
    if batch_param_cls is None:
        raise RuntimeError("qbruntime.BatchParam is unavailable but batch_params were provided")

    normalized = []
    for idx, item in enumerate(batch_params):
        if isinstance(item, batch_param_cls):
            normalized.append(item)
            continue

        if isinstance(item, SequenceBatchParam):
            sequence_length = item.sequence_length
            cache_size = item.cache_size
            cache_id = item.cache_id
        elif isinstance(item, dict):
            sequence_length = item.get("sequence_length")
            cache_size = item.get("cache_size", 0)
            cache_id = item.get("cache_id", idx)
        else:
            sequence_length = getattr(item, "sequence_length", None)
            cache_size = getattr(item, "cache_size", 0)
            cache_id = getattr(item, "cache_id", idx)

        sequence_length = parse_non_negative_int(sequence_length, f"batch_params[{idx}].sequence_length")
        cache_size = parse_non_negative_int(cache_size, f"batch_params[{idx}].cache_size")
        cache_id = parse_non_negative_int(cache_id, f"batch_params[{idx}].cache_id")
        if sequence_length <= 0:
            raise ValueError(f"batch_params[{idx}].sequence_length must be > 0")

        normalized.append(batch_param_cls(sequence_length, cache_size, cache_id))
    return normalized

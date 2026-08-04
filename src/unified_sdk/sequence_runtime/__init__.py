"""
unified_sdk.sequence_runtime
----------------------------
Low-level sequence runtime capability for vendor APIs that expose cache-aware
or batch-parameterized infer primitives.

In the QB-only worktree, this package wraps the Mobilint ARISE Batch LLM style
runtime path separately from the common vision runtime API.
"""

from .api import (
    create_sequence_runtime,
    destroy_sequence_runtime,
    infer_sequence,
    describe_sequence_runtime_api_mapping,
)

# Adapter auto-registration
from . import qb_sequence_runtime as _qb  # noqa: F401

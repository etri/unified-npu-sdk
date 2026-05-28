"""
Unified build entrypoints (backend-agnostic)

This package provides backend-independent model compilation interfaces.
Each backend (e.g., TensorRT, Rebellions, Furiosa) registers its build adapter
in the registry at import time.
"""

from .api import build_unified  # Re-export high-level API

# Internal adapters (auto-registration)
# from . import tensorrt_build as _tensorrt  # noqa: F401
# from . import rbln_build as _rbln          # noqa: F401


import warnings

try:
    from . import tensorrt_build as _tensorrt  # noqa: F401
except Exception as e:
    warnings.warn(f"TensorRT backend disabled: {e!r}")
    _tensorrt = None

try:
    from . import rbln_build as _rbln  # noqa: F401
except Exception as e:
    warnings.warn(f"RBLN backend disabled: {e!r}")
    _rbln = None

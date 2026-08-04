"""
unified_sdk.frontends
---------------------
Frontend helpers for normalizing RNGD LLM model references before they reach
the Furiosa build adapter.
"""

from .resolve_rngd_build_request import (
    describe_frontend_api_mapping,
    detect_prebuilt_artifact_dir,
    resolve_rngd_build_request,
)
from .types import RNGDFrontendBuildRequest, ResolvedRNGDBuildRequest

__all__ = [
    "RNGDFrontendBuildRequest",
    "ResolvedRNGDBuildRequest",
    "describe_frontend_api_mapping",
    "detect_prebuilt_artifact_dir",
    "resolve_rngd_build_request",
]

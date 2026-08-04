"""
unified_sdk.frontends
---------------------
Frontend and prepare modules for normalizing vendor-specific build inputs
before they reach the QB compiler adapter.
"""

from .fetch_qb_artifact import place_provided_qb_artifact
from .export_qb_onnx import export_supported_onnx_from_pth, prepare_supported_module_from_pth, unwrap_state_dict
from .prepare_qb_source import prepare_qb_build_input
from .qb_model_zoo import (
    find_local_mxq,
    find_model_zoo_mxq,
    list_model_zoo_models,
    normalize_mxq_into_models,
    trigger_model_zoo_fetch,
)
from .resolve_qb_build_request import describe_frontend_api_mapping, resolve_qb_build_request
from .types import (
    PreparedQBBuildInput,
    PreparedQBCompileSource,
    ProvidedQBArtifact,
    QBFrontendBuildRequest,
    ResolvedQBBuildRequest,
)

__all__ = [
    "PreparedQBBuildInput",
    "PreparedQBCompileSource",
    "QBFrontendBuildRequest",
    "ResolvedQBBuildRequest",
    "describe_frontend_api_mapping",
    "ProvidedQBArtifact",
    "export_supported_onnx_from_pth",
    "find_local_mxq",
    "find_model_zoo_mxq",
    "list_model_zoo_models",
    "normalize_mxq_into_models",
    "place_provided_qb_artifact",
    "prepare_supported_module_from_pth",
    "prepare_qb_build_input",
    "resolve_qb_build_request",
    "trigger_model_zoo_fetch",
    "unwrap_state_dict",
]

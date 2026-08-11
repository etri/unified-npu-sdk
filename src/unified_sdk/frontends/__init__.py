"""
unified_sdk.frontends
---------------------
Merged frontend exports for backend-specific prepare/fetch flows.

The branch-local resolver implementations are kept as intact as possible and
are re-exported here as the single package surface for `main`.
"""

from .export_qb_onnx import export_supported_onnx_from_pth, prepare_supported_module_from_pth, unwrap_state_dict
from .fetch_qb_artifact import place_provided_qb_artifact
from .prepare_qb_source import prepare_qb_build_input
from .prepare_warboy_runtime_input import inspect_warboy_input_contract, prepare_warboy_runtime_input
from .prepare_warboy_source import prepare_warboy_build_input
from .qb_model_zoo import (
    find_local_mxq,
    find_model_zoo_mxq,
    list_model_zoo_models,
    normalize_mxq_into_models,
    trigger_model_zoo_fetch,
)
from .resolve_qb_build_request import describe_frontend_api_mapping as describe_qb_frontend_api_mapping
from .resolve_qb_build_request import resolve_qb_build_request
from .resolve_rngd_build_request import (
    describe_frontend_api_mapping as describe_rngd_frontend_api_mapping,
    detect_prebuilt_artifact_dir,
    resolve_rngd_build_request,
)
from .resolve_warboy_build_request import describe_frontend_api_mapping as describe_warboy_frontend_api_mapping
from .resolve_warboy_build_request import resolve_warboy_build_request
from .rbln_frontend_impl import list_model_zoo_targets as list_rbln_model_zoo_targets
from .rbln_frontend_impl import prepare_rbln_vision_build_input, resolve_rbln_llm_build_request, resolve_rbln_vision_build_request
from .tensorrt_frontend_impl import (
    classify_tensorrt_llm_source,
    list_torchvision_model_zoo_targets,
    prepare_tensorrt_vision_build_input,
    resolve_tensorrt_llm_build_request,
    resolve_tensorrt_llm_fetch_request,
    resolve_tensorrt_vision_build_request,
)
from .types import (
    PreparedQBBuildInput,
    PreparedQBCompileSource,
    PreparedRBLNCompileSource,
    PreparedRBLNLLMBuildInput,
    PreparedRBLNVisionBuildInput,
    PreparedTensorRTCompileSource,
    PreparedTensorRTLLMBuildInput,
    PreparedTensorRTLLMFetchInput,
    PreparedTensorRTVisionBuildInput,
    PreparedWarboyBuildInput,
    PreparedWarboyCompileSource,
    PreparedWarboyRuntimeInput,
    ProvidedQBArtifact,
    ProvidedRBLNArtifact,
    ProvidedTensorRTArtifact,
    ProvidedWarboyArtifact,
    QBFrontendBuildRequest,
    RBLNLLMFrontendBuildRequest,
    RBLNVisionFrontendBuildRequest,
    RNGDFrontendBuildRequest,
    ResolvedQBBuildRequest,
    ResolvedRBLNLLMBuildRequest,
    ResolvedRBLNVisionBuildRequest,
    ResolvedRNGDBuildRequest,
    ResolvedTensorRTLLMBuildRequest,
    ResolvedTensorRTLLMFetchRequest,
    ResolvedTensorRTVisionBuildRequest,
    ResolvedWarboyBuildRequest,
    TensorRTLLMFrontendBuildRequest,
    TensorRTLLMFrontendFetchRequest,
    TensorRTVisionFrontendBuildRequest,
    WarboyFrontendBuildRequest,
)
from .warboy_model_zoo import fetch_model_zoo_enf, find_local_enf, list_model_zoo_targets, resolve_model_zoo_target


def describe_rbln_frontend_api_mapping():
    return {
        "vision": "resolve_rbln_vision_build_request(request)",
        "llm": "resolve_rbln_llm_build_request(request)",
        "vision_helpers": {
            "prepare": "prepare_rbln_vision_build_input(model_or_path, rbln_path, ...)",
            "model_zoo": "list_rbln_model_zoo_targets()",
        },
    }


def describe_tensorrt_frontend_api_mapping():
    return {
        "vision": "resolve_tensorrt_vision_build_request(request)",
        "llm_fetch": "resolve_tensorrt_llm_fetch_request(request)",
        "llm_build": "resolve_tensorrt_llm_build_request(request)",
        "llm_helpers": {
            "classify_source": "classify_tensorrt_llm_source(model_ref)",
            "vision_prepare": "prepare_tensorrt_vision_build_input(model_or_path, engine_path, ...)",
        },
    }


__all__ = [
    "PreparedQBBuildInput",
    "PreparedQBCompileSource",
    "PreparedRBLNCompileSource",
    "PreparedRBLNLLMBuildInput",
    "PreparedRBLNVisionBuildInput",
    "PreparedTensorRTCompileSource",
    "PreparedTensorRTLLMBuildInput",
    "PreparedTensorRTLLMFetchInput",
    "PreparedTensorRTVisionBuildInput",
    "PreparedWarboyBuildInput",
    "PreparedWarboyCompileSource",
    "PreparedWarboyRuntimeInput",
    "ProvidedQBArtifact",
    "ProvidedRBLNArtifact",
    "ProvidedTensorRTArtifact",
    "ProvidedWarboyArtifact",
    "QBFrontendBuildRequest",
    "RBLNLLMFrontendBuildRequest",
    "RBLNVisionFrontendBuildRequest",
    "RNGDFrontendBuildRequest",
    "ResolvedQBBuildRequest",
    "ResolvedRBLNLLMBuildRequest",
    "ResolvedRBLNVisionBuildRequest",
    "ResolvedRNGDBuildRequest",
    "ResolvedTensorRTLLMBuildRequest",
    "ResolvedTensorRTLLMFetchRequest",
    "ResolvedTensorRTVisionBuildRequest",
    "ResolvedWarboyBuildRequest",
    "TensorRTLLMFrontendBuildRequest",
    "TensorRTLLMFrontendFetchRequest",
    "TensorRTVisionFrontendBuildRequest",
    "WarboyFrontendBuildRequest",
    "classify_tensorrt_llm_source",
    "describe_qb_frontend_api_mapping",
    "describe_rngd_frontend_api_mapping",
    "describe_rbln_frontend_api_mapping",
    "describe_tensorrt_frontend_api_mapping",
    "describe_warboy_frontend_api_mapping",
    "detect_prebuilt_artifact_dir",
    "export_supported_onnx_from_pth",
    "fetch_model_zoo_enf",
    "find_local_enf",
    "find_local_mxq",
    "find_model_zoo_mxq",
    "inspect_warboy_input_contract",
    "list_model_zoo_models",
    "list_model_zoo_targets",
    "list_rbln_model_zoo_targets",
    "list_torchvision_model_zoo_targets",
    "normalize_mxq_into_models",
    "place_provided_qb_artifact",
    "prepare_qb_build_input",
    "prepare_rbln_vision_build_input",
    "prepare_supported_module_from_pth",
    "prepare_tensorrt_vision_build_input",
    "prepare_warboy_build_input",
    "prepare_warboy_runtime_input",
    "resolve_model_zoo_target",
    "resolve_qb_build_request",
    "resolve_rbln_llm_build_request",
    "resolve_rbln_vision_build_request",
    "resolve_rngd_build_request",
    "resolve_tensorrt_llm_build_request",
    "resolve_tensorrt_llm_fetch_request",
    "resolve_tensorrt_vision_build_request",
    "resolve_warboy_build_request",
    "trigger_model_zoo_fetch",
    "unwrap_state_dict",
]

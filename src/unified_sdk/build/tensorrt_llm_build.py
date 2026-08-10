from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from unified_sdk.build.registry import register_llm
from unified_sdk.frontends import resolve_tensorrt_llm_build_request
from unified_sdk.frontends.types import PreparedTensorRTLLMBuildInput, TensorRTLLMFrontendBuildRequest
from unified_sdk.options import resolve_tensorrt_llm_build_options
from unified_sdk.types import BuildResult, LLMBuildConfig


_CAPABILITY_FAMILY = "llm.high-level-generate-builder"
_BUILD_PIPELINE = (
    "resolve_prepared_input",
    "validate_llm_build_options",
    "optional_hf_or_local_model_fetch",
    "optional_tensorrt_llm_compile",
    "optional_artifact_save",
    "emit_metadata",
)
_VENDOR_API_MAP = {
    "fetch": "model ref / local model path passthrough",
    "compile": "tensorrt_llm.LLM(model=..., ...)",
    "save_artifact": "llm.save(engine_dir) [if supported by installed TensorRT-LLM]",
    "artifact": "TensorRT-LLM engine dir",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "model ref / local model path passthrough": "build_unified_LLM(cfg) with fetch contract",
    "tensorrt_llm.LLM(model=..., ...)": "build_unified_LLM(cfg)",
    "llm.save(engine_dir)": "build_unified_LLM(cfg) with artifact_build contract",
    "TensorRT-LLM engine dir": "BuildResult.compiled_model_path",
}


def describe_api_mapping() -> Dict[str, Any]:
    return {
        "unified_api": "build_unified_LLM(cfg)",
        "backend": "tensorrt",
        "capability_family": _CAPABILITY_FAMILY,
        "mapping_direction": "vendor_api ==> unified_api",
        "pipeline": _BUILD_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "vendor_to_unified_api_map": _VENDOR_TO_UNIFIED_API_MAP,
    }


def _ensure_positive_int(value: int, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"LLMBuildConfig.{field_name} must be a positive integer")
    return value


def _best_effort_close(obj: Any) -> None:
    for name in ("shutdown", "close", "dispose"):
        fn = getattr(obj, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def _normalize_llm_kwargs(cfg: LLMBuildConfig, options, prepared_input: PreparedTensorRTLLMBuildInput) -> Dict[str, Any]:
    llm_kwargs: Dict[str, Any] = {
        "model": prepared_input.model_ref,
        "tensor_parallel_size": cfg.tensor_parallel_size,
        "max_seq_len": cfg.max_model_len,
    }
    if options.tokenizer_path:
        llm_kwargs["tokenizer"] = str(options.tokenizer_path)
    if options.dtype:
        llm_kwargs["dtype"] = options.dtype
    if options.trust_remote_code is not None:
        llm_kwargs["trust_remote_code"] = bool(options.trust_remote_code)
    return llm_kwargs


def _compat_prepared_input(cfg: LLMBuildConfig, build_mode: str) -> PreparedTensorRTLLMBuildInput:
    request = TensorRTLLMFrontendBuildRequest(
        model_ref=cfg.model_or_path,
        out_dir=Path(cfg.out_dir),
        model_name=cfg.model_name,
        build_mode=build_mode,
    )
    return resolve_tensorrt_llm_build_request(request).prepared_input


class _TensorRTLLMBuildAdapter:
    name = "tensorrt"

    def build(self, cfg: LLMBuildConfig) -> BuildResult:
        if cfg.backend != self.name:
            raise ValueError(f"TensorRT-LLM build adapter received backend={cfg.backend!r}")

        _ensure_positive_int(cfg.max_model_len, "max_model_len")
        _ensure_positive_int(cfg.tensor_parallel_size, "tensor_parallel_size")
        options = resolve_tensorrt_llm_build_options(cfg.backend_options, extra=dict(cfg.extra or {}))
        prepared_input = cfg.prepared_input or _compat_prepared_input(cfg, options.build_mode)

        if prepared_input.kind == "runtime_model_ref":
            return BuildResult(
                backend=self.name,
                compiled_model_path=prepared_input.model_ref,
                meta_data={
                    "backend": self.name,
                    "track": "llm",
                    "prepared_kind": prepared_input.kind,
                    "build_mode": options.build_mode,
                    "model_ref": prepared_input.model_ref,
                    "backend_options": options.to_metadata(),
                    "capability_family": _CAPABILITY_FAMILY,
                    "build_pipeline": _BUILD_PIPELINE,
                    "vendor_api_map": _VENDOR_API_MAP,
                },
            )

        if prepared_input.kind != "artifact_build" or prepared_input.artifact_dir is None:
            raise ValueError("PreparedTensorRTLLMBuildInput.kind='artifact_build' requires artifact_dir")

        try:
            from tensorrt_llm import LLM
        except Exception as exc:
            raise RuntimeError(
                "tensorrt_llm is required for TensorRT-LLM compile/build. Install it in the container or host env first."
            ) from exc

        if not hasattr(LLM, "save"):
            raise RuntimeError(
                "TensorRT-LLM artifact_build is unsupported in this environment because the installed LLM class "
                "does not expose save(engine_dir). Use fetch mode or provide an already prepared local artifact dir."
            )

        compiled_dir = prepared_input.artifact_dir.expanduser().resolve()
        compiled_dir.parent.mkdir(parents=True, exist_ok=True)
        llm_kwargs = _normalize_llm_kwargs(cfg, options, prepared_input)

        llm = None
        try:
            llm = LLM(**llm_kwargs)
            llm.save(str(compiled_dir))
        except Exception as exc:
            raise RuntimeError(f"TensorRT-LLM compile/save failed for {prepared_input.model_ref}: {exc}") from exc
        finally:
            if llm is not None:
                _best_effort_close(llm)

        return BuildResult(
            backend=self.name,
            compiled_model_path=str(compiled_dir),
            meta_data={
                "backend": self.name,
                "track": "llm",
                "prepared_kind": prepared_input.kind,
                "build_mode": options.build_mode,
                "model_ref": prepared_input.model_ref,
                "compiled_dir": str(compiled_dir),
                "tensor_parallel_size": cfg.tensor_parallel_size,
                "max_model_len": cfg.max_model_len,
                "llm_kwargs": llm_kwargs,
                "backend_options": options.to_metadata(),
                "capability_family": _CAPABILITY_FAMILY,
                "build_pipeline": _BUILD_PIPELINE,
                "vendor_api_map": _VENDOR_API_MAP,
            },
        )


register_llm(_TensorRTLLMBuildAdapter())

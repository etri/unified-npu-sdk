from __future__ import annotations

import subprocess
from typing import Any, Dict

from unified_sdk.build.registry import register_llm
from unified_sdk.frontends.types import PreparedTensorRTLLMBuildInput
from unified_sdk.options import resolve_tensorrt_llm_build_options
from unified_sdk.types import BuildResult, LLMBuildConfig


_CAPABILITY_FAMILY = "llm.multi-phase-fetch-compile-builder"
_BUILD_PIPELINE = (
    "resolve_prepared_input",
    "validate_llm_build_options",
    "classify_fetch_or_compile_variant",
    "optional_model_ref_passthrough",
    "optional_tensorrt_llm_compile",
    "optional_trtllm_build_checkpoint_compile",
    "emit_metadata",
)
_VENDOR_API_MAP = {
    "fetch": "model ref / local model path passthrough",
    "compile_model_ref": "tensorrt_llm.LLM(model=..., ...)",
    "compile_checkpoint_dir": "trtllm-build --checkpoint_dir ... --output_dir ...",
    "save_artifact": "llm.save(engine_dir) [when Python API compile surface is available]",
    "artifact": "TensorRT-LLM engine dir",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "model ref / local model path passthrough": "build_unified_LLM(cfg) with fetch contract",
    "tensorrt_llm.LLM(model=..., ...)": "build_unified_LLM(cfg) for custom_compile(model ref/local path)",
    "trtllm-build --checkpoint_dir ... --output_dir ...": "build_unified_LLM(cfg) for custom_compile(checkpoint dir)",
    "llm.save(engine_dir)": "BuildResult.compiled_model_path for Python API compile",
    "TensorRT-LLM engine dir": "BuildResult.compiled_model_path",
}

_PHASE_SEMANTICS = {
    "fetch": {
        "resolved_phase": "fetch_contract_only",
        "artifact_emitted": False,
        "runtime_may_trigger_vendor_build": True,
    },
    "custom_compile": {
        "resolved_phase": "custom_compile_artifact",
        "artifact_emitted": True,
        "runtime_may_trigger_vendor_build": False,
    },
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
        "tensor_parallel_size": options.tensor_parallel_size,
        "max_seq_len": options.max_model_len,
    }
    if options.tokenizer_path:
        llm_kwargs["tokenizer"] = str(options.tokenizer_path)
    if options.dtype:
        llm_kwargs["dtype"] = options.dtype
    if options.trust_remote_code is not None:
        llm_kwargs["trust_remote_code"] = bool(options.trust_remote_code)
    return llm_kwargs


def _run_trtllm_build(prepared_input: PreparedTensorRTLLMBuildInput, compiled_dir, options) -> None:
    checkpoint_dir = prepared_input.checkpoint_dir
    if checkpoint_dir is None:
        raise ValueError("checkpoint_dir_cli compile requires PreparedTensorRTLLMBuildInput.checkpoint_dir")

    cmd = [
        "trtllm-build",
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--output_dir",
        str(compiled_dir),
        "--max_seq_len",
        str(options.max_model_len),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "TensorRT-LLM checkpoint compile requires the `trtllm-build` CLI in PATH. "
            "Use the official TensorRT-LLM release container or install the CLI first."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or str(exc)
        raise RuntimeError(f"TensorRT-LLM checkpoint compile failed for {checkpoint_dir}: {detail}") from exc


def _validate_prepared_build_mode(prepared_input: PreparedTensorRTLLMBuildInput, build_mode: str) -> None:
    expected = "fetch" if prepared_input.kind == "runtime_model_ref" else "custom_compile"
    if build_mode != expected:
        raise ValueError(
            f"Prepared TensorRT-LLM contract kind={prepared_input.kind!r} requires build_mode={expected!r}, "
            f"but backend_options.build_mode was {build_mode!r}."
        )


def _validate_checkpoint_cli_option_semantics(prepared_input: PreparedTensorRTLLMBuildInput, options) -> None:
    if (prepared_input.compile_variant or "model_ref_api") != "checkpoint_dir_cli":
        return
    unsupported = []
    if options.tokenizer_path is not None:
        unsupported.append("tokenizer_path")
    if options.tensor_parallel_size != 1:
        unsupported.append("tensor_parallel_size")
    if options.dtype is not None:
        unsupported.append("dtype")
    if bool(options.trust_remote_code):
        unsupported.append("trust_remote_code")
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(
            "TensorRT-LLM checkpoint-dir CLI compile only applies max_model_len from TensorRTLLMBuildOptions. "
            f"The following options are not authoritative for checkpoint_dir_cli and must be omitted/defaulted: {joined}."
        )

class _TensorRTLLMBuildAdapter:
    name = "tensorrt"

    def build(self, cfg: LLMBuildConfig) -> BuildResult:
        if cfg.backend != self.name:
            raise ValueError(f"TensorRT-LLM build adapter received backend={cfg.backend!r}")

        options = resolve_tensorrt_llm_build_options(cfg.backend_options)
        prepared_input = cfg.prepared_input
        if prepared_input is None:
            if options.build_mode == "fetch":
                prepared_input = PreparedTensorRTLLMBuildInput(
                    kind="runtime_model_ref",
                    model_ref=str(cfg.model_or_path),
                    source_kind="local_model_path" if str(cfg.model_or_path).startswith(("/", "./", "../", "artifacts/", "build_output/", "models/")) else "model_id",
                    source_path=None,
                    artifact_dir=None,
                )
            else:
                raise ValueError(
                    "TensorRT-LLM artifact build requires a prepared frontend contract. "
                    "Use resolve_tensorrt_llm_build_request(...) and pass LLMBuildConfig.prepared_input."
                )

        _validate_prepared_build_mode(prepared_input, options.build_mode)
        _validate_checkpoint_cli_option_semantics(prepared_input, options)

        if prepared_input.kind == "runtime_model_ref":
            return BuildResult(
                backend=self.name,
                compiled_model_path=prepared_input.model_ref,
                meta_data={
                    "backend": self.name,
                    "track": "llm",
                    "prepared_kind": prepared_input.kind,
                    "source_kind": prepared_input.source_kind,
                    "source_path": str(prepared_input.source_path) if prepared_input.source_path is not None else None,
                    "build_mode": options.build_mode,
                    "model_ref": prepared_input.model_ref,
                    "backend_options": options.to_metadata(),
                    "capability_family": _CAPABILITY_FAMILY,
                    "build_pipeline": _BUILD_PIPELINE,
                    "vendor_api_map": _VENDOR_API_MAP,
                    **_PHASE_SEMANTICS["fetch"],
                },
            )

        if prepared_input.kind != "artifact_build" or prepared_input.artifact_dir is None:
            raise ValueError("PreparedTensorRTLLMBuildInput.kind='artifact_build' requires artifact_dir")

        compiled_dir = prepared_input.artifact_dir.expanduser().resolve()
        compiled_dir.parent.mkdir(parents=True, exist_ok=True)
        compile_variant = prepared_input.compile_variant or "model_ref_api"
        llm_kwargs = _normalize_llm_kwargs(cfg, options, prepared_input)

        if compile_variant == "checkpoint_dir_cli":
            _run_trtllm_build(prepared_input, compiled_dir, options)
        else:
            try:
                from tensorrt_llm import LLM
            except Exception as exc:
                raise RuntimeError(
                    "tensorrt_llm is required for TensorRT-LLM custom compile/build. "
                    "Install it in the container or host env first."
                ) from exc
            llm = None
            try:
                llm = LLM(**llm_kwargs)
                if not hasattr(llm, "save"):
                    raise RuntimeError(
                        "the installed LLM class does not expose save(engine_dir). "
                        "For custom compile, prefer a local TensorRT-LLM checkpoint dir so the SDK can use `trtllm-build`."
                    )
                llm.save(str(compiled_dir))
            except Exception as exc:
                raise RuntimeError(f"TensorRT-LLM custom compile failed for {prepared_input.model_ref}: {exc}") from exc
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
                "source_kind": prepared_input.source_kind,
                "source_path": str(prepared_input.source_path) if prepared_input.source_path is not None else None,
                "build_mode": options.build_mode,
                "model_ref": prepared_input.model_ref,
                "compiled_dir": str(compiled_dir),
                "compile_variant": compile_variant,
                "checkpoint_dir": str(prepared_input.checkpoint_dir) if prepared_input.checkpoint_dir is not None else None,
                "tensor_parallel_size": options.tensor_parallel_size,
                "max_model_len": options.max_model_len,
                "llm_kwargs": llm_kwargs,
                "backend_options": options.to_metadata(),
                "capability_family": _CAPABILITY_FAMILY,
                "build_pipeline": _BUILD_PIPELINE,
                "vendor_api_map": _VENDOR_API_MAP,
                **_PHASE_SEMANTICS["custom_compile"],
            },
        )


register_llm(_TensorRTLLMBuildAdapter())

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from unified_sdk.build.registry import register_llm
from unified_sdk.frontends import PreparedRBLNLLMBuildInput
from unified_sdk.options import resolve_rbln_llm_build_options
from unified_sdk.types import BuildResult, LLMBuildConfig


_CAPABILITY_FAMILY = "llm.high-level-compiler-and-artifact"
_BUILD_PIPELINE = (
    "validate_config",
    "select_build_mode",
    "fetch_model_ref_or_compile_with_optimum",
    "save_pretrained_artifact_if_needed",
    "emit_metadata",
)
_VENDOR_API_MAP = {
    "fetch": "model id or local model path passed through to downstream LLM runtime",
    "optimum_compile": "optimum.rbln.RBLNAutoModelForCausalLM.from_pretrained(..., export=True)",
    "save_artifact": "compiled_model.save_pretrained(output_dir)",
    "artifact": "precompiled RBLN HuggingFace-style directory",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "model id or local model path passed through to downstream LLM runtime": "build_unified_LLM(cfg) for runtime model-ref passthrough",
    "RBLNAutoModelForCausalLM.from_pretrained(..., export=True)": "build_unified_LLM(cfg) for artifact build via optimum-rbln",
    "compiled_model.save_pretrained(output_dir)": "BuildResult.compiled_model_path",
}


def _require_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"LLMBuildConfig.{field_name} must be a positive integer, got {value!r}")
    return value


def _artifact_dir(out_dir: str | Path, model_name: str) -> Path:
    name = str(model_name).strip()
    if not name:
        raise ValueError("LLMBuildConfig.model_name must be a non-empty string")
    return Path(out_dir) / name


def describe_api_mapping() -> Dict[str, Any]:
    return {
        "unified_api": "build_unified_LLM(cfg)",
        "backend": "rbln",
        "capability_family": _CAPABILITY_FAMILY,
        "mapping_direction": "vendor_api ==> unified_api",
        "pipeline": _BUILD_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "vendor_to_unified_api_map": _VENDOR_TO_UNIFIED_API_MAP,
    }


def _coerce_prepared_input(cfg: LLMBuildConfig, mode: str) -> PreparedRBLNLLMBuildInput:
    if cfg.prepared_input is not None:
        return cfg.prepared_input

    model_ref = str(cfg.model_or_path).strip()
    if mode == "fetch":
        return PreparedRBLNLLMBuildInput(
            kind="runtime_model_ref",
            model_ref=model_ref,
            artifact_dir=None,
        )

    raise RuntimeError(
        "RBLN LLM artifact build now expects a prepared frontend contract. "
        "Call resolve_rbln_llm_build_request(...) first and pass "
        "LLMBuildConfig(prepared_input=...)."
    )


class _RBLNLLMBuildAdapter:
    name = "rbln"

    def build(self, cfg: LLMBuildConfig) -> BuildResult:
        if cfg.backend != self.name:
            raise ValueError(f"RBLN LLM build adapter received backend={cfg.backend!r}")

        options = resolve_rbln_llm_build_options(cfg.backend_options, extra=dict(cfg.extra or {}))
        options_meta = options.to_metadata()
        mode = options.build_mode
        prepared_input = _coerce_prepared_input(cfg, mode)
        model_ref = prepared_input.model_ref

        if prepared_input.kind == "runtime_model_ref":
            if mode != "fetch":
                raise RuntimeError(
                    "Prepared runtime-model-ref input requires build_mode='fetch'. "
                    f"Received build_mode={mode!r}."
                )
            return BuildResult(
                backend=self.name,
                compiled_model_path=model_ref,
                meta_data={
                    "backend": self.name,
                    "source": "runtime_model_ref",
                    "model_ref": model_ref,
                    "artifact_emitted": False,
                    "capability_variant": "runtime_model_ref_passthrough",
                    "note": "model id or local model path; loaded by the selected LLM runtime implementation",
                    "backend_options": options_meta,
                    "capability_family": _CAPABILITY_FAMILY,
                    "build_pipeline": _BUILD_PIPELINE,
                    "vendor_api_map": _VENDOR_API_MAP,
                    "selected_path": "model_ref",
                    "prepared_kind": prepared_input.kind,
                    "build_mode": mode,
                },
            )

        if mode != "optimum_compile":
            raise RuntimeError(
                "Prepared artifact-build input requires build_mode='optimum_compile'. "
                f"Received build_mode={mode!r}."
            )

        _require_positive_int(cfg.batch_size, "batch_size")
        _require_positive_int(cfg.max_model_len, "max_model_len")
        _require_positive_int(cfg.num_devices, "num_devices")

        artifact_dir = prepared_input.artifact_dir or _artifact_dir(cfg.out_dir, cfg.model_name)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        try:
            from optimum.rbln import RBLNAutoModelForCausalLM
        except Exception as exc:
            raise RuntimeError(
                "optimum-rbln is required for RBLN LLM custom compile. "
                "Install optimum-rbln first (see docs.rbln.ai)."
            ) from exc

        compile_kwargs: Dict[str, Any] = {
            "export": True,
            "rbln_batch_size": cfg.batch_size,
            "rbln_max_seq_len": cfg.max_model_len,
            "rbln_num_devices": cfg.num_devices,
        }
        compile_kwargs["trust_remote_code"] = options.trust_remote_code
        if options.revision:
            compile_kwargs["revision"] = options.revision
        compile_kwargs["rbln_create_runtimes"] = options.rbln_create_runtimes

        try:
            compiled = RBLNAutoModelForCausalLM.from_pretrained(model_ref, **compile_kwargs)
            compiled.save_pretrained(str(artifact_dir))
        except Exception as exc:
            raise RuntimeError(f"RBLN LLM optimum compile failed: {exc}") from exc

        return BuildResult(
            backend=self.name,
            compiled_model_path=str(artifact_dir),
            meta_data={
                "backend": self.name,
                "source": "optimum_rbln",
                "model_ref": model_ref,
                "artifact_dir": str(artifact_dir),
                "artifact_emitted": True,
                "capability_variant": "artifact_build",
                "batch_size": cfg.batch_size,
                "max_model_len": cfg.max_model_len,
                "num_devices": cfg.num_devices,
                "backend_options": options_meta,
                "capability_family": _CAPABILITY_FAMILY,
                "build_pipeline": _BUILD_PIPELINE,
                "vendor_api_map": _VENDOR_API_MAP,
                "selected_path": "optimum_compile",
                "prepared_kind": prepared_input.kind,
                "build_mode": mode,
            },
        )


def build_llm(cfg: LLMBuildConfig) -> BuildResult:
    return _RBLNLLMBuildAdapter().build(cfg)


register_llm(_RBLNLLMBuildAdapter())

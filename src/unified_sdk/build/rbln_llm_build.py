from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

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
    "model id or local model path passed through to downstream LLM runtime": "build_unified_LLM(cfg) when extra['build_mode'] is absent or 'fetch'",
    "RBLNAutoModelForCausalLM.from_pretrained(..., export=True)": "build_unified_LLM(cfg) when extra['build_mode'] == 'optimum_compile'",
    "compiled_model.save_pretrained(output_dir)": "BuildResult.compiled_model_path",
}


def _require_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"LLMBuildConfig.{field_name} must be a positive integer, got {value!r}")
    return value


def _normalize_build_mode(extra: Dict[str, Any]) -> str:
    mode = str(extra.get("build_mode", "fetch")).strip().lower()
    if mode not in {"fetch", "optimum_compile"}:
        raise ValueError(f"Unsupported RBLN LLM build mode: {mode!r}")
    return mode


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


def build_llm(cfg: LLMBuildConfig) -> BuildResult:
    if cfg.backend != "rbln":
        raise ValueError(f"RBLN LLM build adapter received backend={cfg.backend!r}")

    extra = dict(cfg.extra or {})
    mode = _normalize_build_mode(extra)
    model_ref = str(cfg.model_or_path)

    if mode == "fetch":
        return BuildResult(
            backend="rbln",
            compiled_model_path=model_ref,
            meta_data={
                "backend": "rbln",
                "source": "provided",
                "model_ref": model_ref,
                "note": "model id or local model path; loaded by the selected LLM runtime implementation",
                "extra": extra,
                "capability_family": _CAPABILITY_FAMILY,
                "build_pipeline": _BUILD_PIPELINE,
                "vendor_api_map": _VENDOR_API_MAP,
                "selected_path": "model_ref",
                "build_mode": mode,
            },
        )

    _require_positive_int(cfg.batch_size, "batch_size")
    _require_positive_int(cfg.max_model_len, "max_model_len")
    _require_positive_int(cfg.num_devices, "num_devices")

    artifact_dir = _artifact_dir(cfg.out_dir, cfg.model_name)
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
    if "trust_remote_code" in extra:
        compile_kwargs["trust_remote_code"] = bool(extra["trust_remote_code"])
    if "revision" in extra and extra["revision"]:
        compile_kwargs["revision"] = str(extra["revision"])
    if "rbln_create_runtimes" in extra:
        compile_kwargs["rbln_create_runtimes"] = bool(extra["rbln_create_runtimes"])
    else:
        compile_kwargs["rbln_create_runtimes"] = False

    try:
        compiled = RBLNAutoModelForCausalLM.from_pretrained(model_ref, **compile_kwargs)
        compiled.save_pretrained(str(artifact_dir))
    except Exception as exc:
        raise RuntimeError(f"RBLN LLM optimum compile failed: {exc}") from exc

    return BuildResult(
        backend="rbln",
        compiled_model_path=str(artifact_dir),
        meta_data={
            "backend": "rbln",
            "source": "optimum_rbln",
            "model_ref": model_ref,
            "artifact_dir": str(artifact_dir),
            "batch_size": cfg.batch_size,
            "max_model_len": cfg.max_model_len,
            "num_devices": cfg.num_devices,
            "extra": extra,
            "capability_family": _CAPABILITY_FAMILY,
            "build_pipeline": _BUILD_PIPELINE,
            "vendor_api_map": _VENDOR_API_MAP,
            "selected_path": "optimum_compile",
            "build_mode": mode,
        },
    )

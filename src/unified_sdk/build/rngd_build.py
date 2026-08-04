from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, Dict, List

from unified_sdk.build.registry import register
from unified_sdk.options import RNGDBuildOptions
from unified_sdk.types import BuildConfig, BuildResult


_CAPABILITY_FAMILY = "llm.fxb-and-generation"
_BUILD_PIPELINE = (
    "validate_config",
    "resolve_model_ref",
    "select_build_mode",
    "return_model_ref_or_run_fxb_build",
    "verify_fxb",
    "emit_metadata",
)
_VENDOR_API_MAP = {
    "fetch": "HF model id or local model path passed through to furiosa_llm.LLM(...)",
    "fxb_build": "fxb build <model_id_or_path> <output_path> [options]",
    "parallel_config": "fxb build --tensor-parallel-size / --pipeline-parallel-size",
    "model_config": "fxb build --max-model-len",
    "artifact": "HF model id or .fxb file path",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "HF model id or local model path passed through to furiosa_llm.LLM(...)": "build_unified_LLM(cfg) when backend_options.build_mode is absent or 'fetch'",
    "fxb build <model_id_or_path> <output_path> [options]": "build_unified_LLM(cfg) when backend_options.build_mode == 'fxb_build'",
    "fxb build --tensor-parallel-size / --pipeline-parallel-size": "BuildConfig.tensor_parallel_size / pipeline_parallel_size",
    "fxb build --max-model-len": "BuildConfig.max_model_len",
    "HF model id or .fxb file path": "BuildResult.compiled_model_path",
}


def _require_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"BuildConfig.{field_name} must be a positive integer, got {value!r}")
    return value


def _fxb_output_path(out_dir: Path, model_name: str) -> Path:
    output = out_dir / model_name
    if output.suffix != ".fxb":
        output = output.with_suffix(".fxb")
    return output


def _detect_prebuilt_artifact_dir(model_ref: str) -> str | None:
    p = Path(model_ref)
    if not p.is_dir():
        return None

    markers = []
    for name in ("artifact.json", "binary_bundle.zip", "model_metadata.json"):
        if (p / name).exists():
            markers.append(name)

    if not markers:
        return None
    return ", ".join(markers)


def _looks_like_qwen3_8b_fp8(model_ref: str, model_name: str) -> bool:
    text = f"{model_ref} {model_name}".lower().replace("_", "-")
    return "qwen3-8b-fp8" in text


def _capability_metadata(options: RNGDBuildOptions, source: str) -> Dict[str, Any]:
    return {
        "capability_family": _CAPABILITY_FAMILY,
        "build_pipeline": _BUILD_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "selected_path": source,
        **options.to_metadata(),
    }


def describe_api_mapping() -> Dict[str, Any]:
    return {
        "unified_api": "build_unified_LLM(cfg)",
        "backend": "rngd",
        "capability_family": _CAPABILITY_FAMILY,
        "mapping_direction": "vendor_api ==> unified_api",
        "pipeline": _BUILD_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "vendor_to_unified_api_map": _VENDOR_TO_UNIFIED_API_MAP,
    }


class _RNGDBuildAdapter:
    """FuriosaAI RNGD build adapter.

    공식 smoke 기준은 fetch + LLM.generate 경로다.
    선택적으로 FXB 기반 custom model build 경로를 지원한다.

      1) fetch (기본): HF 모델 id 또는 로컬 모델 경로를 그대로 사용한다.
      2) fxb_build: `fxb build`로 .fxb 를 생성한다.
    """

    name = "rngd"

    def build(self, cfg: BuildConfig) -> BuildResult:
        if cfg.backend != self.name:
            raise ValueError(f"RNGD build adapter received backend={cfg.backend!r}")

        extra = dict(cfg.extra or {})
        options = RNGDBuildOptions.from_raw(cfg.backend_options, legacy_extra=extra)
        model_ref = str(cfg.model_or_path)
        mode = options.build_mode

        if mode == "fetch":
            meta: Dict[str, Any] = {
                "backend": self.name,
                "source": "provided",
                "model_ref": model_ref,
                "note": "HF model id or local model path; loaded by furiosa_llm.LLM at runtime",
                "backend_options": options.to_metadata(),
                "extra": extra,
                **_capability_metadata(options, "model_ref"),
            }
            return BuildResult(
                backend=self.name,
                compiled_model_path=model_ref,
                meta_data=meta,
            )

        tp = _require_positive_int(cfg.tensor_parallel_size, "tensor_parallel_size")
        pp = _require_positive_int(cfg.pipeline_parallel_size, "pipeline_parallel_size")
        artifact_markers = _detect_prebuilt_artifact_dir(model_ref)
        if artifact_markers:
            raise RuntimeError(
                "FXB build expects an upstream/raw Hugging Face model snapshot or a local model directory, "
                f"but {model_ref!r} looks like a prebuilt Furiosa artifact repo ({artifact_markers}). "
                "Use this path with the standard smoke (`model id/local artifact -> generate`) instead, "
                "or prepare an upstream model snapshot such as 'Qwen/Qwen3-8B-FP8' for custom FXB smoke."
            )

        if _looks_like_qwen3_8b_fp8(model_ref, cfg.model_name) and tp == 1:
            raise RuntimeError(
                "FXB build for Qwen3-8B-FP8 with tensor_parallel_size=1 is vendor-confirmed as unsupported. "
                "Use tensor_parallel_size=8 for RNGD 1-card smoke, or follow the model-specific TP combinations "
                "documented by FuriosaAI."
            )

        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = _fxb_output_path(out_dir, cfg.model_name)

        cmd: List[str] = [
            "fxb",
            "build",
            model_ref,
            str(output_path),
            "--tensor-parallel-size",
            str(tp),
            "--pipeline-parallel-size",
            str(pp),
        ]
        if cfg.max_model_len:
            cmd.extend(["--max-model-len", str(int(cfg.max_model_len))])
        if options.optim_level:
            cmd.extend(["--optim-level", str(options.optim_level)])
        if options.dry_run:
            cmd.append("--dry-run")
        if options.build_report:
            cmd.append("--build-report")
        if options.concurrency:
            cmd.extend(["--concurrency", str(options.concurrency)])

        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "The `fxb` command was not found. Official FXB build requires furiosa-llm with FXB support."
            ) from exc

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            detail = stderr or stdout or f"`fxb build` failed with exit code {proc.returncode}"
            if (
                _looks_like_qwen3_8b_fp8(model_ref, cfg.model_name)
                and tp == 1
                and "tcc subprocess failed" in detail
            ):
                detail += (
                    " | Hint: FuriosaAI confirmed on July 14, 2026 that Qwen3-8B-FP8 does not support TP=1. "
                    "Retry with --tensor-parallel-size 8 for RNGD 1-card smoke."
                )
            raise RuntimeError(f"FXB build failed: {detail}")

        if not options.dry_run and not output_path.is_file():
            raise RuntimeError(f"`fxb build` reported success but FXB file not found at {output_path}")

        meta = {
            "backend": self.name,
            "source": "fxb_build",
            "model_ref": model_ref,
            "fxb_path": str(output_path),
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp,
            "max_model_len": cfg.max_model_len,
            "command": cmd,
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(),
            "backend_options": options.to_metadata(),
            "extra": extra,
            **_capability_metadata(options, "fxb_build"),
        }
        return BuildResult(
            backend=self.name,
            compiled_model_path=str(output_path),
            meta_data=meta,
        )


register(_RNGDBuildAdapter())

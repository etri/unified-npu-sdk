from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

from unified_sdk.build.registry import register
from unified_sdk.frontends.types import (
    PreparedWarboyBuildInput,
    PreparedWarboyCompileSource,
    ProvidedWarboyArtifact,
)
from unified_sdk.options import WarboyBuildOptions, resolve_warboy_build_options
from unified_sdk.types import BuildConfig, BuildResult


_CAPABILITY_FAMILY = "vision.cli-compiler"
_BUILD_PIPELINE = (
    "validate_config",
    "resolve_artifact_or_quantized_onnx",
    "resolve_compiler_options",
    "run_vendor_compiler_cli_or_place_artifact",
    "verify_artifact",
    "emit_metadata",
)
_VENDOR_API_MAP = {
    "provided_artifact": "frontend prepare/fetch resolves a normalized .enf path -> shutil.copyfile(src_enf, enf_path)",
    "compile": "furiosa-compiler <quantized_onnx> -o <enf_path> --target-npu <target_npu> --target-ir enf",
    "prepare": "prepare_warboy_quantized_onnx.py or equivalent frontend helper before build_unified(cfg)",
    "artifact": ".enf",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "frontend resolve step": "run_warboy_build.py / frontend helper before build_unified(cfg)",
    "shutil.copyfile(src_enf, enf_path)": "build_unified(cfg) for provided .enf",
    "furiosa-compiler <quantized_onnx> -o <enf_path> ...": "build_unified(cfg) for quantized ONNX compile",
    "prepare_warboy_quantized_onnx.py": "prepare quantized ONNX before build_unified(cfg)",
    ".enf artifact": "BuildResult.compiled_model_path",
}


def _require_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"BuildConfig.{field_name} must be a non-empty string")
    return value.strip()


def _validate_shape(shape: Tuple[int, ...], field_name: str) -> Tuple[int, ...]:
    if not isinstance(shape, tuple) or not shape:
        raise ValueError(f"BuildConfig.{field_name} must be a non-empty tuple of positive integers")
    if not all(isinstance(dim, int) and dim > 0 for dim in shape):
        raise ValueError(f"BuildConfig.{field_name} must contain only positive integers: {shape!r}")
    return shape


def _build_output_path(out_dir: str | Path, model_name: str) -> Path:
    name = _require_non_empty_string(model_name, "model_name")
    path = Path(out_dir) / name
    if path.suffix != ".enf":
        path = path.with_suffix(".enf")
    return path


def _resolve_model_source(model_or_path: str | Path) -> tuple[str, Path]:
    source_path = Path(model_or_path).expanduser().resolve()
    suffix = source_path.suffix.lower()
    if suffix == ".enf":
        return "provided", source_path
    if suffix == ".onnx":
        return "quantized_onnx", source_path
    raise ValueError(
        "Warboy build expects either a provided .enf artifact or a quantized .onnx compile source. "
        f"Got: {source_path}"
    )


def _coerce_prepared_input(cfg: BuildConfig, enf_path: Path) -> PreparedWarboyBuildInput:
    if cfg.prepared_input is not None:
        return cfg.prepared_input

    source_kind, source_path = _resolve_model_source(cfg.model_or_path)
    if source_kind == "provided":
        return PreparedWarboyBuildInput(
            kind="provided_artifact",
            provided_artifact=ProvidedWarboyArtifact(
                source_path=source_path,
                destination_path=enf_path,
            ),
            compile_source=None,
        )
    return PreparedWarboyBuildInput(
        kind="compile_source",
        compile_source=PreparedWarboyCompileSource(
            source=str(source_path),
            source_label="quantized_onnx",
        ),
        provided_artifact=None,
    )


def _capability_metadata(options: WarboyBuildOptions, source: str) -> Dict[str, Any]:
    return {
        "capability_family": _CAPABILITY_FAMILY,
        "build_pipeline": _BUILD_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "selected_path": source,
        "compile_options": options.to_metadata(),
    }


def describe_api_mapping() -> Dict[str, Any]:
    return {
        "unified_api": "build_unified(cfg)",
        "backend": "warboy",
        "capability_family": _CAPABILITY_FAMILY,
        "mapping_direction": "vendor_api ==> unified_api",
        "pipeline": _BUILD_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "vendor_to_unified_api_map": _VENDOR_TO_UNIFIED_API_MAP,
    }


class _WarboyBuildAdapter:
    """FuriosaAI Warboy build adapter.

    두 가지 경로를 지원한다:
      1) 이미 컴파일된 .enf 를 제공받은 경우 -> out_dir 로 배치 (provided/fetch)
      2) quantized ONNX 를 받은 경우         -> furiosa-compiler 로 .enf 컴파일 (compile)

    양자화(f32 ONNX -> quantized ONNX)는 build core가 아니라 prepare capability로 본다.
    """

    name = "warboy"

    def build(self, cfg: BuildConfig) -> BuildResult:
        if cfg.backend != self.name:
            raise ValueError(f"Warboy build adapter received backend={cfg.backend!r}")

        options = resolve_warboy_build_options(cfg.backend_options)
        enf_path = _build_output_path(cfg.out_dir, cfg.model_name)
        enf_path.parent.mkdir(parents=True, exist_ok=True)
        prepared_input = _coerce_prepared_input(cfg, enf_path)

        if prepared_input.kind == "provided_artifact":
            artifact = prepared_input.provided_artifact
            if artifact is None:
                raise RuntimeError("Warboy prepared_input.kind='provided_artifact' requires provided_artifact payload")
            src = artifact.source_path
            if not src.is_file():
                raise FileNotFoundError(f"Provided .enf not found: {src}")
            if src != enf_path:
                shutil.copyfile(src, enf_path)
            meta: Dict[str, Any] = {
                "backend": self.name,
                "enf_path": str(enf_path),
                "source": "provided",
                "origin": str(src),
                "precision": "int8",
                "backend_options": options.to_metadata(),
                **_capability_metadata(options, "provided"),
            }
            return BuildResult(
                backend=self.name,
                compiled_model_path=str(enf_path),
                meta_data=meta,
            )

        compile_source = prepared_input.compile_source
        if compile_source is None:
            raise RuntimeError("Warboy prepared_input.kind='compile_source' requires compile_source payload")

        onnx_path = Path(compile_source.source).expanduser().resolve()
        if not onnx_path.is_file():
            raise FileNotFoundError(f"quantized ONNX not found: {onnx_path}")

        compiler = shutil.which("furiosa-compiler")
        if compiler is None:
            raise RuntimeError(
                "furiosa-compiler not found on PATH. Install the FuriosaAI Warboy toolchain first "
                "(APT: furiosa-compiler; see developer.furiosa.ai)."
            )

        if cfg.input_shape is not None:
            _validate_shape(tuple(cfg.input_shape), "input_shape")
        command = [
            compiler,
            str(onnx_path),
            "-o",
            str(enf_path),
            "--target-npu",
            options.target_npu,
            "--target-ir",
            options.target_ir,
        ]
        if options.compiler_config:
            command.extend(str(arg) for arg in options.compiler_config)

        try:
            completed = subprocess.run(command, check=False, text=True, capture_output=True)
        except Exception as exc:
            raise RuntimeError(f"furiosa-compiler invocation failed: {exc}") from exc
        if completed.returncode != 0:
            raise RuntimeError(
                f"furiosa-compiler failed (exit {completed.returncode}):\n{completed.stdout}\n{completed.stderr}"
            )

        if not enf_path.is_file():
            raise RuntimeError(f"furiosa-compiler reported success but .enf not found at {enf_path}")

        meta = {
            "backend": self.name,
            "enf_path": str(enf_path),
            "source": "furiosa_compiler",
            "target_npu": options.target_npu,
            "target_ir": options.target_ir,
            "onnx_path": str(onnx_path),
            "input_shape": tuple(cfg.input_shape) if cfg.input_shape is not None else None,
            "precision": "int8",
            "backend_options": options.to_metadata(),
            **_capability_metadata(options, "furiosa_compiler"),
        }
        return BuildResult(
            backend=self.name,
            compiled_model_path=str(enf_path),
            meta_data=meta,
        )


register(_WarboyBuildAdapter())

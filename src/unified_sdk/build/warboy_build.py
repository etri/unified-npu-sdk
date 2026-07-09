from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

from unified_sdk.build.registry import register
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
    "provided_artifact": "shutil.copyfile(src_enf, enf_path)",
    "compile": "furiosa-compiler <quantized_onnx> -o <enf_path> --target-npu <target_npu> --target-ir enf",
    "pre_quantization": "furiosa.quantizer (not wrapped here; quantized ONNX is expected)",
    "artifact": ".enf",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "shutil.copyfile(src_enf, enf_path)": "build_unified(cfg) for provided .enf",
    "furiosa-compiler <quantized_onnx> -o <enf_path> ...": "build_unified(cfg) for quantized ONNX compile",
    "furiosa.quantizer": "not wrapped; prepare quantized ONNX before build_unified(cfg)",
    ".enf artifact": "BuildResult.compiled_model_path",
}


# furiosa-compiler 타깃 (Warboy 는 2 PE. 참조 가이드 기준 warboy-2pe 사용)
_TARGET_NPUS = ("warboy", "warboy-2pe")


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


def _validate_extra(extra: Dict[str, Any]) -> Dict[str, Any]:
    target_npu = extra.get("target_npu")
    if target_npu is not None and target_npu not in _TARGET_NPUS:
        raise ValueError(
            "BuildConfig.extra['target_npu'] must be one of: " + ", ".join(repr(t) for t in _TARGET_NPUS)
        )
    target_ir = extra.get("target_ir")
    if target_ir is not None and (not isinstance(target_ir, str) or not target_ir.strip()):
        raise ValueError("BuildConfig.extra['target_ir'] must be a non-empty string when provided")
    return extra


def _build_output_path(out_dir: str | Path, model_name: str) -> Path:
    name = _require_non_empty_string(model_name, "model_name")
    path = Path(out_dir) / name
    if path.suffix != ".enf":
        path = path.with_suffix(".enf")
    return path


def _looks_like_enf(model_or_path: Any) -> bool:
    return isinstance(model_or_path, (str, Path)) and str(model_or_path).endswith(".enf")


def _capability_metadata(extra: Dict[str, Any], source: str) -> Dict[str, Any]:
    return {
        "capability_family": _CAPABILITY_FAMILY,
        "build_pipeline": _BUILD_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "selected_path": source,
        "compile_options": {
            "target_npu": extra.get("target_npu", "warboy-2pe"),
            "target_ir": extra.get("target_ir", "enf"),
            "compiler_config": extra.get("compiler_config"),
        },
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

    두 가지 경로를 지원한다 (fetch 기본 + compile 훅):
      1) 이미 컴파일된 .enf 를 제공받은 경우  -> 검증 후 out_dir 로 배치 (provided/fetch)
      2) quantized ONNX 를 받은 경우          -> furiosa-compiler 로 .enf 컴파일 (compile hook)

    양자화(f32 ONNX -> quantized ONNX)는 furiosa.quantizer 로 별도 수행한다
    (host_validation_tools 및 참조 가이드 9.3A 참고).
    """

    name = "warboy"

    def build(self, cfg: BuildConfig) -> BuildResult:
        if cfg.backend != self.name:
            raise ValueError(f"Warboy build adapter received backend={cfg.backend!r}")

        if cfg.precision not in ("int8",):
            raise ValueError(f"Unsupported Warboy precision: {cfg.precision!r} (ENF is int8-quantized)")

        extra = _validate_extra(dict(cfg.extra or {}))
        enf_path = _build_output_path(cfg.out_dir, cfg.model_name)
        enf_path.parent.mkdir(parents=True, exist_ok=True)

        # ---- Path 1: 사전 컴파일된 .enf 제공 (fetch / provided) ----
        if _looks_like_enf(cfg.model_or_path):
            src = Path(cfg.model_or_path).expanduser().resolve()
            if not src.is_file():
                raise FileNotFoundError(f"Provided .enf not found: {src}")
            if src != enf_path.resolve():
                shutil.copyfile(src, enf_path)
            meta: Dict[str, Any] = {
                "backend": self.name,
                "enf_path": str(enf_path),
                "source": "provided",
                "origin": str(src),
                "precision": cfg.precision,
                "extra": extra,
                **_capability_metadata(extra, "provided"),
            }
            return BuildResult(
                backend=self.name,
                compiled_model_path=str(enf_path),
                meta_data=meta,
            )

        # ---- Path 2: furiosa-compiler 로 quantized ONNX -> .enf (compile hook) ----
        onnx_path = Path(cfg.model_or_path).expanduser().resolve()
        if not onnx_path.is_file():
            raise FileNotFoundError(f"quantized ONNX not found: {onnx_path}")

        compiler = shutil.which("furiosa-compiler")
        if compiler is None:
            raise RuntimeError(
                "furiosa-compiler not found on PATH. Install the FuriosaAI Warboy toolchain first "
                "(APT: furiosa-compiler; see developer.furiosa.ai)."
            )

        _validate_shape(tuple(cfg.input_shape), "input_shape")
        target_npu = extra.get("target_npu", "warboy-2pe")
        target_ir = extra.get("target_ir", "enf")

        command = [
            compiler,
            str(onnx_path),
            "-o",
            str(enf_path),
            "--target-npu",
            target_npu,
            "--target-ir",
            target_ir,
        ]
        # 선택 컴파일 옵션 패스스루 (예: extra['compiler_config'] = ['--batch-size', '1'])
        compiler_config = extra.get("compiler_config")
        if isinstance(compiler_config, (list, tuple)):
            command.extend(str(a) for a in compiler_config)

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
            "target_npu": target_npu,
            "target_ir": target_ir,
            "onnx_path": str(onnx_path),
            "input_shape": tuple(cfg.input_shape),
            "precision": cfg.precision,
            "extra": extra,
            **_capability_metadata(extra, "furiosa_compiler"),
        }
        return BuildResult(
            backend=self.name,
            compiled_model_path=str(enf_path),
            meta_data=meta,
        )


register(_WarboyBuildAdapter())

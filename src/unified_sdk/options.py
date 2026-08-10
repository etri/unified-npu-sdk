from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal


_VISION_BUILD_LEGACY_KEYS = frozenset(
    {"precision", "workspace_mib", "strict_types", "int8_calibrator"}
)
_VISION_RUNTIME_LEGACY_KEYS = frozenset({"use_execute_v3", "allow_dynamic_shape"})
_LLM_BUILD_LEGACY_KEYS = frozenset(
    {"build_mode", "tokenizer_path", "tensor_parallel_size", "max_model_len", "dtype", "trust_remote_code"}
)
_LLM_RUNTIME_LEGACY_KEYS = frozenset(
    {"tokenizer_path", "tensor_parallel_size", "max_model_len", "dtype", "trust_remote_code"}
)


def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    if value in (0, 1):
        return bool(value)
    raise ValueError(f"{field_name} must be a boolean-like value, got {value!r}")


def _validate_legacy_extra(extra: Dict[str, Any], *, allowed_keys: frozenset[str], option_label: str) -> Dict[str, Any]:
    unknown_keys = sorted(key for key in extra if key not in allowed_keys)
    if unknown_keys:
        joined = ", ".join(repr(key) for key in unknown_keys)
        raise ValueError(
            f"{option_label} legacy extra contains unsupported keys: {joined}. "
            "Pass typed backend_options instead of relying on extra passthrough."
        )
    return dict(extra)


@dataclass(frozen=True)
class TensorRTVisionBuildOptions:
    precision: Literal["fp32", "fp16", "int8"] = "fp32"
    workspace_mib: int | None = None
    strict_types: bool = False
    int8_calibrator: Any = None

    def normalized(self) -> "TensorRTVisionBuildOptions":
        precision = str(self.precision).strip().lower()
        if precision not in {"fp32", "fp16", "int8"}:
            raise ValueError("TensorRTVisionBuildOptions.precision must be 'fp32', 'fp16', or 'int8'")
        workspace_mib = None
        if self.workspace_mib is not None:
            workspace_mib = int(self.workspace_mib)
            if workspace_mib <= 0:
                raise ValueError("TensorRTVisionBuildOptions.workspace_mib must be > 0")
        return TensorRTVisionBuildOptions(
            precision=precision,  # type: ignore[arg-type]
            workspace_mib=workspace_mib,
            strict_types=_parse_bool(self.strict_types, "TensorRTVisionBuildOptions.strict_types"),
            int8_calibrator=self.int8_calibrator,
        )

    def to_metadata(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {
            "precision": normalized.precision,
            "workspace_mib": normalized.workspace_mib,
            "strict_types": normalized.strict_types,
            "int8_calibrator": "<provided>" if normalized.int8_calibrator is not None else None,
        }

    @classmethod
    def from_legacy_extra(cls, extra: Dict[str, Any]) -> "TensorRTVisionBuildOptions":
        extra = _validate_legacy_extra(
            extra,
            allowed_keys=_VISION_BUILD_LEGACY_KEYS,
            option_label="TensorRTVisionBuildOptions",
        )
        return cls(
            precision=extra.get("precision", "fp32"),
            workspace_mib=extra.get("workspace_mib"),
            strict_types=extra.get("strict_types", False),
            int8_calibrator=extra.get("int8_calibrator"),
        ).normalized()


@dataclass(frozen=True)
class TensorRTVisionRuntimeOptions:
    use_execute_v3: bool = True
    allow_dynamic_shape: bool = False

    def normalized(self) -> "TensorRTVisionRuntimeOptions":
        return TensorRTVisionRuntimeOptions(
            use_execute_v3=_parse_bool(self.use_execute_v3, "TensorRTVisionRuntimeOptions.use_execute_v3"),
            allow_dynamic_shape=_parse_bool(
                self.allow_dynamic_shape,
                "TensorRTVisionRuntimeOptions.allow_dynamic_shape",
            ),
        )

    def to_metadata(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {
            "use_execute_v3": normalized.use_execute_v3,
            "allow_dynamic_shape": normalized.allow_dynamic_shape,
        }

    @classmethod
    def from_legacy_extra(cls, extra: Dict[str, Any]) -> "TensorRTVisionRuntimeOptions":
        extra = _validate_legacy_extra(
            extra,
            allowed_keys=_VISION_RUNTIME_LEGACY_KEYS,
            option_label="TensorRTVisionRuntimeOptions",
        )
        return cls(
            use_execute_v3=extra.get("use_execute_v3", True),
            allow_dynamic_shape=extra.get("allow_dynamic_shape", False),
        ).normalized()


@dataclass(frozen=True)
class TensorRTLLMBuildOptions:
    build_mode: Literal["fetch", "llm_api_compile"] = "fetch"
    tokenizer_path: str | Path | None = None
    tensor_parallel_size: int = 1
    max_model_len: int = 512
    dtype: str | None = None
    trust_remote_code: bool = False

    def normalized(self) -> "TensorRTLLMBuildOptions":
        build_mode = str(self.build_mode).strip().lower()
        if build_mode not in {"fetch", "llm_api_compile"}:
            raise ValueError("TensorRTLLMBuildOptions.build_mode must be 'fetch' or 'llm_api_compile'")
        tensor_parallel_size = int(self.tensor_parallel_size)
        if tensor_parallel_size <= 0:
            raise ValueError("TensorRTLLMBuildOptions.tensor_parallel_size must be > 0")
        max_model_len = int(self.max_model_len)
        if max_model_len <= 0:
            raise ValueError("TensorRTLLMBuildOptions.max_model_len must be > 0")
        return TensorRTLLMBuildOptions(
            build_mode=build_mode,  # type: ignore[arg-type]
            tokenizer_path=_normalize_optional_str(self.tokenizer_path),
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype=_normalize_optional_str(self.dtype),
            trust_remote_code=_parse_bool(
                self.trust_remote_code,
                "TensorRTLLMBuildOptions.trust_remote_code",
            ),
        )

    def to_metadata(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {
            "build_mode": normalized.build_mode,
            "tokenizer_path": str(normalized.tokenizer_path) if normalized.tokenizer_path is not None else None,
            "tensor_parallel_size": normalized.tensor_parallel_size,
            "max_model_len": normalized.max_model_len,
            "dtype": normalized.dtype,
            "trust_remote_code": normalized.trust_remote_code,
        }

    @classmethod
    def from_legacy_extra(cls, extra: Dict[str, Any]) -> "TensorRTLLMBuildOptions":
        extra = _validate_legacy_extra(
            extra,
            allowed_keys=_LLM_BUILD_LEGACY_KEYS,
            option_label="TensorRTLLMBuildOptions",
        )
        return cls(
            build_mode=extra.get("build_mode", "fetch"),
            tokenizer_path=extra.get("tokenizer_path"),
            tensor_parallel_size=extra.get("tensor_parallel_size", 1),
            max_model_len=extra.get("max_model_len", 512),
            dtype=extra.get("dtype"),
            trust_remote_code=extra.get("trust_remote_code", False),
        ).normalized()


@dataclass(frozen=True)
class TensorRTLLMRuntimeOptions:
    tokenizer_path: str | Path | None = None
    tensor_parallel_size: int = 1
    max_model_len: int = 512
    dtype: str | None = None
    trust_remote_code: bool = False

    def normalized(self) -> "TensorRTLLMRuntimeOptions":
        tensor_parallel_size = int(self.tensor_parallel_size)
        if tensor_parallel_size <= 0:
            raise ValueError("TensorRTLLMRuntimeOptions.tensor_parallel_size must be > 0")
        max_model_len = int(self.max_model_len)
        if max_model_len <= 0:
            raise ValueError("TensorRTLLMRuntimeOptions.max_model_len must be > 0")
        return TensorRTLLMRuntimeOptions(
            tokenizer_path=_normalize_optional_str(self.tokenizer_path),
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype=_normalize_optional_str(self.dtype),
            trust_remote_code=_parse_bool(
                self.trust_remote_code,
                "TensorRTLLMRuntimeOptions.trust_remote_code",
            ),
        )

    def to_metadata(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {
            "tokenizer_path": str(normalized.tokenizer_path) if normalized.tokenizer_path is not None else None,
            "tensor_parallel_size": normalized.tensor_parallel_size,
            "max_model_len": normalized.max_model_len,
            "dtype": normalized.dtype,
            "trust_remote_code": normalized.trust_remote_code,
        }

    @classmethod
    def from_legacy_extra(cls, extra: Dict[str, Any]) -> "TensorRTLLMRuntimeOptions":
        extra = _validate_legacy_extra(
            extra,
            allowed_keys=_LLM_RUNTIME_LEGACY_KEYS,
            option_label="TensorRTLLMRuntimeOptions",
        )
        return cls(
            tokenizer_path=extra.get("tokenizer_path"),
            tensor_parallel_size=extra.get("tensor_parallel_size", 1),
            max_model_len=extra.get("max_model_len", 512),
            dtype=extra.get("dtype"),
            trust_remote_code=extra.get("trust_remote_code", False),
        ).normalized()


def resolve_tensorrt_vision_build_options(options: Any) -> TensorRTVisionBuildOptions:
    if isinstance(options, TensorRTVisionBuildOptions):
        return options.normalized()
    if options is None:
        return TensorRTVisionBuildOptions().normalized()
    raise TypeError("BuildConfig.backend_options must be a TensorRTVisionBuildOptions instance when provided")


def resolve_tensorrt_vision_runtime_options(options: Any) -> TensorRTVisionRuntimeOptions:
    if isinstance(options, TensorRTVisionRuntimeOptions):
        return options.normalized()
    if options is None:
        return TensorRTVisionRuntimeOptions().normalized()
    raise TypeError("RuntimeConfig.backend_options must be a TensorRTVisionRuntimeOptions instance when provided")


def resolve_tensorrt_llm_build_options(options: Any) -> TensorRTLLMBuildOptions:
    if isinstance(options, TensorRTLLMBuildOptions):
        return options.normalized()
    if options is None:
        return TensorRTLLMBuildOptions().normalized()
    raise TypeError("LLMBuildConfig.backend_options must be a TensorRTLLMBuildOptions instance when provided")


def resolve_tensorrt_llm_runtime_options(options: Any) -> TensorRTLLMRuntimeOptions:
    if isinstance(options, TensorRTLLMRuntimeOptions):
        return options.normalized()
    if options is None:
        return TensorRTLLMRuntimeOptions().normalized()
    raise TypeError("LLMRuntimeConfig.backend_options must be a TensorRTLLMRuntimeOptions instance when provided")

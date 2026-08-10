from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal


_VISION_FRONTENDS = ("rebel", "optimum_image_classification")
_TRACE_METHODS = ("export", "export_strict", "jittrace")
_TENSOR_TYPES = ("np", "pt")
_VISION_BUILD_LEGACY_KEYS = frozenset({"npu", "precision", "model_trace_method"})
_VISION_RUNTIME_LEGACY_KEYS = frozenset({"device", "tensor_type", "timeout", "activate_profiler", "allow_dynamic_shape"})
_LLM_BUILD_LEGACY_KEYS = frozenset({"build_mode", "trust_remote_code", "revision", "rbln_create_runtimes"})
_LLM_RUNTIME_LEGACY_KEYS = frozenset(
    {
        "runtime_impl",
        "tensor_parallel_size",
        "max_model_len",
        "block_size",
        "trust_remote_code",
        "enforce_eager",
        "dtype",
        "gpu_memory_utilization",
        "additional_config",
    }
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
class RBLNVisionBuildOptions:
    npu: str | None = None
    precision: Literal["fp32", "fp16"] = "fp16"
    model_trace_method: str | None = None

    def normalized(self) -> "RBLNVisionBuildOptions":
        npu = _normalize_optional_str(self.npu)
        precision = str(self.precision).strip().lower()
        if precision not in {"fp32", "fp16"}:
            raise ValueError("RBLNVisionBuildOptions.precision must be 'fp32' or 'fp16'")
        model_trace_method = _normalize_optional_str(self.model_trace_method)
        if model_trace_method is not None and model_trace_method not in _TRACE_METHODS:
            raise ValueError(
                "RBLNVisionBuildOptions.model_trace_method must be one of: "
                + ", ".join(repr(item) for item in _TRACE_METHODS)
            )
        return RBLNVisionBuildOptions(
            npu=npu,
            precision=precision,  # type: ignore[arg-type]
            model_trace_method=model_trace_method,
        )

    def to_metadata(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {
            "npu": normalized.npu,
            "precision": normalized.precision,
            "model_trace_method": normalized.model_trace_method,
        }

    @classmethod
    def from_legacy_extra(cls, extra: Dict[str, Any]) -> "RBLNVisionBuildOptions":
        extra = _validate_legacy_extra(
            extra,
            allowed_keys=_VISION_BUILD_LEGACY_KEYS,
            option_label="RBLNVisionBuildOptions",
        )
        return cls(
            npu=extra.get("npu"),
            precision=extra.get("precision", "fp16"),
            model_trace_method=extra.get("model_trace_method"),
        ).normalized()


@dataclass(frozen=True)
class RBLNVisionRuntimeOptions:
    device: int = 0
    tensor_type: Literal["np", "pt"] = "np"
    timeout: float | None = None
    activate_profiler: bool = False
    allow_dynamic_shape: bool = False

    def normalized(self) -> "RBLNVisionRuntimeOptions":
        try:
            device = int(self.device)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"RBLNVisionRuntimeOptions.device must be an integer, got {self.device!r}") from exc
        if device < 0:
            raise ValueError("RBLNVisionRuntimeOptions.device must be >= 0")
        tensor_type = str(self.tensor_type).strip()
        if tensor_type not in _TENSOR_TYPES:
            raise ValueError(
                "RBLNVisionRuntimeOptions.tensor_type must be one of: "
                + ", ".join(repr(item) for item in _TENSOR_TYPES)
            )
        timeout = None
        if self.timeout is not None:
            timeout = float(self.timeout)
            if timeout <= 0:
                raise ValueError("RBLNVisionRuntimeOptions.timeout must be > 0")
        return RBLNVisionRuntimeOptions(
            device=device,
            tensor_type=tensor_type,  # type: ignore[arg-type]
            timeout=timeout,
            activate_profiler=_parse_bool(self.activate_profiler, "RBLNVisionRuntimeOptions.activate_profiler"),
            allow_dynamic_shape=_parse_bool(self.allow_dynamic_shape, "RBLNVisionRuntimeOptions.allow_dynamic_shape"),
        )

    def to_metadata(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {
            "device": normalized.device,
            "tensor_type": normalized.tensor_type,
            "timeout": normalized.timeout,
            "activate_profiler": normalized.activate_profiler,
            "allow_dynamic_shape": normalized.allow_dynamic_shape,
        }

    @classmethod
    def from_legacy_extra(cls, extra: Dict[str, Any]) -> "RBLNVisionRuntimeOptions":
        extra = _validate_legacy_extra(
            extra,
            allowed_keys=_VISION_RUNTIME_LEGACY_KEYS,
            option_label="RBLNVisionRuntimeOptions",
        )
        return cls(
            device=extra.get("device", 0),
            tensor_type=extra.get("tensor_type", "np"),
            timeout=extra.get("timeout"),
            activate_profiler=extra.get("activate_profiler", False),
            allow_dynamic_shape=extra.get("allow_dynamic_shape", False),
        ).normalized()


@dataclass(frozen=True)
class RBLNLLMBuildOptions:
    build_mode: Literal["fetch", "optimum_compile"] = "fetch"
    trust_remote_code: bool = False
    revision: str | None = None
    rbln_create_runtimes: bool = False

    def normalized(self) -> "RBLNLLMBuildOptions":
        build_mode = str(self.build_mode).strip().lower()
        if build_mode not in {"fetch", "optimum_compile"}:
            raise ValueError("RBLNLLMBuildOptions.build_mode must be 'fetch' or 'optimum_compile'")
        return RBLNLLMBuildOptions(
            build_mode=build_mode,  # type: ignore[arg-type]
            trust_remote_code=_parse_bool(self.trust_remote_code, "RBLNLLMBuildOptions.trust_remote_code"),
            revision=_normalize_optional_str(self.revision),
            rbln_create_runtimes=_parse_bool(
                self.rbln_create_runtimes,
                "RBLNLLMBuildOptions.rbln_create_runtimes",
            ),
        )

    def to_metadata(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {
            "build_mode": normalized.build_mode,
            "trust_remote_code": normalized.trust_remote_code,
            "revision": normalized.revision,
            "rbln_create_runtimes": normalized.rbln_create_runtimes,
        }

    @classmethod
    def from_legacy_extra(cls, extra: Dict[str, Any]) -> "RBLNLLMBuildOptions":
        extra = _validate_legacy_extra(
            extra,
            allowed_keys=_LLM_BUILD_LEGACY_KEYS,
            option_label="RBLNLLMBuildOptions",
        )
        return cls(
            build_mode=extra.get("build_mode", "fetch"),
            trust_remote_code=extra.get("trust_remote_code", False),
            revision=extra.get("revision"),
            rbln_create_runtimes=extra.get("rbln_create_runtimes", False),
        ).normalized()


@dataclass(frozen=True)
class RBLNLLMRuntimeOptions:
    runtime_impl: Literal["vllm"] = "vllm"
    tensor_parallel_size: int = 1
    max_model_len: int = 512
    block_size: int | None = None
    trust_remote_code: bool = False
    enforce_eager: bool = False
    dtype: str | None = None
    gpu_memory_utilization: float | None = None
    additional_config: Dict[str, Any] | None = None

    def normalized(self) -> "RBLNLLMRuntimeOptions":
        runtime_impl = str(self.runtime_impl).strip().lower()
        if runtime_impl != "vllm":
            raise ValueError("RBLNLLMRuntimeOptions.runtime_impl currently supports only 'vllm'")
        tensor_parallel_size = int(self.tensor_parallel_size)
        if tensor_parallel_size <= 0:
            raise ValueError("RBLNLLMRuntimeOptions.tensor_parallel_size must be > 0")
        max_model_len = int(self.max_model_len)
        if max_model_len <= 0:
            raise ValueError("RBLNLLMRuntimeOptions.max_model_len must be > 0")
        block_size = None if self.block_size is None else int(self.block_size)
        if block_size is not None and block_size <= 0:
            raise ValueError("RBLNLLMRuntimeOptions.block_size must be > 0 when provided")
        gpu_memory_utilization = None
        if self.gpu_memory_utilization is not None:
            gpu_memory_utilization = float(self.gpu_memory_utilization)
        additional_config = None
        if self.additional_config is not None:
            if not isinstance(self.additional_config, dict):
                raise ValueError("RBLNLLMRuntimeOptions.additional_config must be a dict when provided")
            additional_config = dict(self.additional_config)
        return RBLNLLMRuntimeOptions(
            runtime_impl=runtime_impl,  # type: ignore[arg-type]
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            block_size=block_size,
            trust_remote_code=_parse_bool(self.trust_remote_code, "RBLNLLMRuntimeOptions.trust_remote_code"),
            enforce_eager=_parse_bool(self.enforce_eager, "RBLNLLMRuntimeOptions.enforce_eager"),
            dtype=_normalize_optional_str(self.dtype),
            gpu_memory_utilization=gpu_memory_utilization,
            additional_config=additional_config,
        )

    def to_metadata(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {
            "runtime_impl": normalized.runtime_impl,
            "tensor_parallel_size": normalized.tensor_parallel_size,
            "max_model_len": normalized.max_model_len,
            "block_size": normalized.block_size,
            "trust_remote_code": normalized.trust_remote_code,
            "enforce_eager": normalized.enforce_eager,
            "dtype": normalized.dtype,
            "gpu_memory_utilization": normalized.gpu_memory_utilization,
            "additional_config": dict(normalized.additional_config) if normalized.additional_config is not None else None,
        }

    @classmethod
    def from_legacy_extra(cls, extra: Dict[str, Any]) -> "RBLNLLMRuntimeOptions":
        extra = _validate_legacy_extra(
            extra,
            allowed_keys=_LLM_RUNTIME_LEGACY_KEYS,
            option_label="RBLNLLMRuntimeOptions",
        )
        return cls(
            runtime_impl=extra.get("runtime_impl", "vllm"),
            tensor_parallel_size=extra.get("tensor_parallel_size", 1),
            max_model_len=extra.get("max_model_len", 512),
            block_size=extra.get("block_size"),
            trust_remote_code=extra.get("trust_remote_code", False),
            enforce_eager=extra.get("enforce_eager", False),
            dtype=extra.get("dtype"),
            gpu_memory_utilization=extra.get("gpu_memory_utilization"),
            additional_config=extra.get("additional_config"),
        ).normalized()


def resolve_rbln_vision_build_options(
    options: Any,
    *,
    precision: str | None = None,
    extra: Dict[str, Any] | None = None,
) -> RBLNVisionBuildOptions:
    if isinstance(options, RBLNVisionBuildOptions):
        normalized = options.normalized()
    elif options is not None:
        raise TypeError("BuildConfig.backend_options must be an RBLNVisionBuildOptions instance when provided")
    else:
        legacy = dict(extra or {})
        if precision is not None:
            legacy["precision"] = precision
        normalized = RBLNVisionBuildOptions.from_legacy_extra(legacy)
    return normalized


def resolve_rbln_vision_runtime_options(options: Any, *, extra: Dict[str, Any] | None = None) -> RBLNVisionRuntimeOptions:
    if isinstance(options, RBLNVisionRuntimeOptions):
        return options.normalized()
    if options is not None:
        raise TypeError("RuntimeConfig.backend_options must be an RBLNVisionRuntimeOptions instance when provided")
    return RBLNVisionRuntimeOptions.from_legacy_extra(extra or {})


def resolve_rbln_llm_build_options(options: Any, *, extra: Dict[str, Any] | None = None) -> RBLNLLMBuildOptions:
    if isinstance(options, RBLNLLMBuildOptions):
        return options.normalized()
    if options is not None:
        raise TypeError("LLMBuildConfig.backend_options must be an RBLNLLMBuildOptions instance when provided")
    return RBLNLLMBuildOptions.from_legacy_extra(extra or {})


def resolve_rbln_llm_runtime_options(options: Any, *, extra: Dict[str, Any] | None = None) -> RBLNLLMRuntimeOptions:
    if isinstance(options, RBLNLLMRuntimeOptions):
        return options.normalized()
    if options is not None:
        raise TypeError("LLMRuntimeConfig.backend_options must be an RBLNLLMRuntimeOptions instance when provided")
    return RBLNLLMRuntimeOptions.from_legacy_extra(extra or {})

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal


_QUANTIZE_METHODS = ("percentile", "maxpercentile", "max", "kl")
_TARGET_DEVICE_BY_PRODUCT = {
    "aries": "aries-rb",
    "aries-rb": "aries-rb",
    "regulus": "regulus-rb",
    "regulus-rb": "regulus-rb",
    "regulus-ra": "regulus-ra",
}
_TARGET_NPUS = ("warboy", "warboy-2pe")
_TRACE_METHODS = ("export", "export_strict", "jittrace")
_TENSOR_TYPES = ("np", "pt")
_VISION_BUILD_LEGACY_KEYS = frozenset(
    {"npu", "precision", "model_trace_method", "workspace_mib", "strict_types", "int8_calibrator"}
)
_VISION_RUNTIME_LEGACY_KEYS = frozenset(
    {"device", "tensor_type", "timeout", "activate_profiler", "allow_dynamic_shape", "use_execute_v3"}
)
_LLM_BUILD_LEGACY_KEYS = frozenset(
    {
        "build_mode",
        "trust_remote_code",
        "revision",
        "rbln_create_runtimes",
        "tokenizer_path",
        "tensor_parallel_size",
        "max_model_len",
        "dtype",
    }
)
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
        "tokenizer_path",
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


def _positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
    return value


def _positive_int_or_none(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


@dataclass(frozen=True)
class QBBuildOptions:
    quantize_method: str = "percentile"
    use_random_calib: bool | None = None
    calib_data_path: str | None = None
    product: str = "aries"
    target_device: str | None = None
    model_nickname: str | None = None
    optimize_option: Any = None
    singlecore_compile: Any = None
    save_sample: Any = None

    def _normalize_in_place(self) -> None:
        if isinstance(self.product, str):
            object.__setattr__(self, "product", self.product.strip())
        if isinstance(self.target_device, str):
            normalized = self.target_device.strip()
            object.__setattr__(self, "target_device", normalized or None)

    def validate(self) -> "QBBuildOptions":
        self._normalize_in_place()
        if self.quantize_method not in _QUANTIZE_METHODS:
            raise ValueError(
                "QBBuildOptions.quantize_method must be one of: "
                + ", ".join(repr(m) for m in _QUANTIZE_METHODS)
            )
        if not isinstance(self.product, str) or not self.product.strip():
            raise ValueError("QBBuildOptions.product must be a non-empty string")
        if self.calib_data_path is not None and (
            not isinstance(self.calib_data_path, str) or not self.calib_data_path.strip()
        ):
            raise ValueError("QBBuildOptions.calib_data_path must be a non-empty string when provided")
        if self.target_device is not None and (
            not isinstance(self.target_device, str) or not self.target_device.strip()
        ):
            raise ValueError("QBBuildOptions.target_device must be a non-empty string when provided")
        return self

    def resolved_target_device(self) -> str:
        if self.target_device:
            return self.target_device.strip()
        product = self.product.strip().lower()
        return _TARGET_DEVICE_BY_PRODUCT.get(product, product)

    def compile_options_metadata(self) -> Dict[str, Any]:
        return {
            "quantize_method": self.quantize_method,
            "use_random_calib": self.use_random_calib,
            "calib_data_path": self.calib_data_path,
            "product": self.product,
            "target_device": self.target_device,
            "model_nickname": self.model_nickname,
            "optimize_option": self.optimize_option,
            "singlecore_compile": self.singlecore_compile,
            "save_sample": self.save_sample,
        }


@dataclass(frozen=True)
class QBVisionRuntimeOptions:
    core_mode: str | None = None
    allow_dynamic_shape: bool = False

    def _normalize_in_place(self) -> None:
        if isinstance(self.core_mode, str):
            normalized = self.core_mode.strip()
            object.__setattr__(self, "core_mode", normalized or None)

    def validate(self) -> "QBVisionRuntimeOptions":
        self._normalize_in_place()
        if self.core_mode is not None and (not isinstance(self.core_mode, str) or not self.core_mode.strip()):
            raise ValueError("QBVisionRuntimeOptions.core_mode must be a non-empty string when provided")
        return self


@dataclass(frozen=True)
class QBSequenceRuntimeOptions:
    core_mode: str | None = None
    allow_dynamic_shape: bool = False

    def _normalize_in_place(self) -> None:
        if isinstance(self.core_mode, str):
            normalized = self.core_mode.strip()
            object.__setattr__(self, "core_mode", normalized or None)

    def validate(self) -> "QBSequenceRuntimeOptions":
        self._normalize_in_place()
        if self.core_mode is not None and (not isinstance(self.core_mode, str) or not self.core_mode.strip()):
            raise ValueError("QBSequenceRuntimeOptions.core_mode must be a non-empty string when provided")
        return self


@dataclass(frozen=True)
class WarboyBuildOptions:
    target_npu: str = "warboy-2pe"
    target_ir: str = "enf"
    compiler_config: tuple[str, ...] = ()

    def normalized(self) -> "WarboyBuildOptions":
        target_npu = str(self.target_npu).strip()
        if target_npu not in _TARGET_NPUS:
            raise ValueError(
                "WarboyBuildOptions.target_npu must be one of: " + ", ".join(repr(t) for t in _TARGET_NPUS)
            )
        target_ir = str(self.target_ir).strip()
        if not target_ir:
            raise ValueError("WarboyBuildOptions.target_ir must be a non-empty string")
        compiler_config = tuple(str(item) for item in self.compiler_config)
        return WarboyBuildOptions(target_npu=target_npu, target_ir=target_ir, compiler_config=compiler_config)

    def to_metadata(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {
            "target_npu": normalized.target_npu,
            "target_ir": normalized.target_ir,
            "compiler_config": list(normalized.compiler_config),
        }


@dataclass(frozen=True)
class WarboyRuntimeOptions:
    device: str | None = None
    allow_dynamic_shape: bool = False

    def normalized(self) -> "WarboyRuntimeOptions":
        return WarboyRuntimeOptions(
            device=_normalize_optional_str(self.device),
            allow_dynamic_shape=_parse_bool(self.allow_dynamic_shape, "WarboyRuntimeOptions.allow_dynamic_shape"),
        )

    def to_metadata(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {"device": normalized.device, "allow_dynamic_shape": normalized.allow_dynamic_shape}


@dataclass(frozen=True)
class RNGDBuildOptions:
    build_mode: Literal["fetch", "fxb_build"] = "fetch"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: int | None = None
    dry_run: bool = False
    optim_level: str | None = None
    build_report: bool = False
    concurrency: int | None = None

    def normalized(self) -> "RNGDBuildOptions":
        mode = str(self.build_mode).strip().lower()
        if mode not in {"fetch", "fxb_build"}:
            raise ValueError(f"Unsupported RNGD build mode: {self.build_mode!r}")
        return RNGDBuildOptions(
            build_mode=mode,
            tensor_parallel_size=_positive_int(self.tensor_parallel_size, "RNGDBuildOptions.tensor_parallel_size"),
            pipeline_parallel_size=_positive_int(
                self.pipeline_parallel_size,
                "RNGDBuildOptions.pipeline_parallel_size",
            ),
            max_model_len=_positive_int_or_none(self.max_model_len, "RNGDBuildOptions.max_model_len"),
            dry_run=bool(self.dry_run),
            optim_level=_normalize_optional_str(self.optim_level),
            build_report=bool(self.build_report),
            concurrency=_positive_int_or_none(self.concurrency, "RNGDBuildOptions.concurrency"),
        )

    def to_metadata(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {
            "build_mode": normalized.build_mode,
            "tensor_parallel_size": normalized.tensor_parallel_size,
            "pipeline_parallel_size": normalized.pipeline_parallel_size,
            "max_model_len": normalized.max_model_len,
            "dry_run": normalized.dry_run,
            "optim_level": normalized.optim_level,
            "build_report": normalized.build_report,
            "concurrency": normalized.concurrency,
        }


@dataclass(frozen=True)
class RNGDRuntimeOptions:
    fxb_path: str | Path | None = None
    devices: str | None = None

    def normalized(self) -> "RNGDRuntimeOptions":
        fxb_path = self.fxb_path
        if isinstance(fxb_path, Path):
            fxb_path = str(fxb_path)
        return RNGDRuntimeOptions(
            fxb_path=_normalize_optional_str(fxb_path),
            devices=_normalize_optional_str(self.devices),
        )

    def to_metadata(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {"fxb_path": normalized.fxb_path, "devices": normalized.devices}


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
        return RBLNVisionBuildOptions(npu=npu, precision=precision, model_trace_method=model_trace_method)

    def to_metadata(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {
            "npu": normalized.npu,
            "precision": normalized.precision,
            "model_trace_method": normalized.model_trace_method,
        }

    @classmethod
    def from_legacy_extra(cls, extra: Dict[str, Any]) -> "RBLNVisionBuildOptions":
        extra = _validate_legacy_extra(extra, allowed_keys=_VISION_BUILD_LEGACY_KEYS, option_label="RBLNVisionBuildOptions")
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
                "RBLNVisionRuntimeOptions.tensor_type must be one of: " + ", ".join(repr(item) for item in _TENSOR_TYPES)
            )
        timeout = None
        if self.timeout is not None:
            timeout = float(self.timeout)
            if timeout <= 0:
                raise ValueError("RBLNVisionRuntimeOptions.timeout must be > 0")
        return RBLNVisionRuntimeOptions(
            device=device,
            tensor_type=tensor_type,
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
        extra = _validate_legacy_extra(extra, allowed_keys=_VISION_RUNTIME_LEGACY_KEYS, option_label="RBLNVisionRuntimeOptions")
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
            build_mode=build_mode,
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
        extra = _validate_legacy_extra(extra, allowed_keys=_LLM_BUILD_LEGACY_KEYS, option_label="RBLNLLMBuildOptions")
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
        gpu_memory_utilization = None if self.gpu_memory_utilization is None else float(self.gpu_memory_utilization)
        additional_config = None
        if self.additional_config is not None:
            if not isinstance(self.additional_config, dict):
                raise ValueError("RBLNLLMRuntimeOptions.additional_config must be a dict when provided")
            additional_config = dict(self.additional_config)
        return RBLNLLMRuntimeOptions(
            runtime_impl=runtime_impl,
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
        extra = _validate_legacy_extra(extra, allowed_keys=_LLM_RUNTIME_LEGACY_KEYS, option_label="RBLNLLMRuntimeOptions")
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
            precision=precision,
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
        extra = _validate_legacy_extra(extra, allowed_keys=_VISION_BUILD_LEGACY_KEYS, option_label="TensorRTVisionBuildOptions")
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
        extra = _validate_legacy_extra(extra, allowed_keys=_VISION_RUNTIME_LEGACY_KEYS, option_label="TensorRTVisionRuntimeOptions")
        return cls(
            use_execute_v3=extra.get("use_execute_v3", True),
            allow_dynamic_shape=extra.get("allow_dynamic_shape", False),
        ).normalized()


@dataclass(frozen=True)
class TensorRTLLMBuildOptions:
    tokenizer_path: str | Path | None = None
    tensor_parallel_size: int = 1
    max_model_len: int = 512
    dtype: str | None = None
    trust_remote_code: bool = False

    def normalized(self) -> "TensorRTLLMBuildOptions":
        tensor_parallel_size = int(self.tensor_parallel_size)
        if tensor_parallel_size <= 0:
            raise ValueError("TensorRTLLMBuildOptions.tensor_parallel_size must be > 0")
        max_model_len = int(self.max_model_len)
        if max_model_len <= 0:
            raise ValueError("TensorRTLLMBuildOptions.max_model_len must be > 0")
        return TensorRTLLMBuildOptions(
            tokenizer_path=_normalize_optional_str(self.tokenizer_path),
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype=_normalize_optional_str(self.dtype),
            trust_remote_code=_parse_bool(self.trust_remote_code, "TensorRTLLMBuildOptions.trust_remote_code"),
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
            trust_remote_code=_parse_bool(self.trust_remote_code, "TensorRTLLMRuntimeOptions.trust_remote_code"),
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


def resolve_qb_build_options(options: Any) -> QBBuildOptions:
    if isinstance(options, QBBuildOptions):
        return options.validate()
    if options is not None:
        raise TypeError("BuildConfig.backend_options must be a QBBuildOptions instance when provided")
    return QBBuildOptions().validate()


def resolve_qb_runtime_options(options: Any) -> QBVisionRuntimeOptions:
    if isinstance(options, QBVisionRuntimeOptions):
        return options.validate()
    if options is not None:
        raise TypeError("RuntimeConfig.backend_options must be a QBVisionRuntimeOptions instance when provided")
    return QBVisionRuntimeOptions().validate()


def resolve_qb_sequence_runtime_options(options: Any) -> QBSequenceRuntimeOptions:
    if isinstance(options, QBSequenceRuntimeOptions):
        return options.validate()
    if options is not None:
        raise TypeError("SequenceRuntimeConfig.backend_options must be a QBSequenceRuntimeOptions instance when provided")
    return QBSequenceRuntimeOptions().validate()


def resolve_warboy_build_options(options: Any) -> WarboyBuildOptions:
    if isinstance(options, WarboyBuildOptions):
        return options.normalized()
    if options is not None:
        raise TypeError("BuildConfig.backend_options must be a WarboyBuildOptions instance when provided")
    return WarboyBuildOptions().normalized()


def resolve_warboy_runtime_options(options: Any) -> WarboyRuntimeOptions:
    if isinstance(options, WarboyRuntimeOptions):
        return options.normalized()
    if options is not None:
        raise TypeError("RuntimeConfig.backend_options must be a WarboyRuntimeOptions instance when provided")
    return WarboyRuntimeOptions().normalized()


def resolve_rngd_build_options(options: Any) -> RNGDBuildOptions:
    if isinstance(options, RNGDBuildOptions):
        return options.normalized()
    if options is not None:
        raise TypeError("LLMBuildConfig.backend_options must be an RNGDBuildOptions instance when provided")
    return RNGDBuildOptions().normalized()


def resolve_rngd_runtime_options(options: Any) -> RNGDRuntimeOptions:
    if isinstance(options, RNGDRuntimeOptions):
        return options.normalized()
    if options is not None:
        raise TypeError("LLMRuntimeConfig.backend_options must be an RNGDRuntimeOptions instance when provided")
    return RNGDRuntimeOptions().normalized()


def resolve_rbln_vision_build_options(options: Any, *, precision: str | None = None, extra: Dict[str, Any] | None = None) -> RBLNVisionBuildOptions:
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

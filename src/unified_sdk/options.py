from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


_QUANTIZE_METHODS = ("percentile", "maxpercentile", "max", "kl")
_TARGET_DEVICE_BY_PRODUCT = {
    "aries": "aries-rb",
    "aries-rb": "aries-rb",
    "regulus": "regulus-rb",
    "regulus-rb": "regulus-rb",
    "regulus-ra": "regulus-ra",
}


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

    @classmethod
    def from_legacy_extra(cls, extra: Mapping[str, Any] | None) -> "QBBuildOptions":
        extra = dict(extra or {})
        return cls(
            quantize_method=str(extra.get("quantize_method", "percentile")),
            use_random_calib=extra.get("use_random_calib"),
            calib_data_path=str(extra["calib_data_path"]).strip() if extra.get("calib_data_path") else None,
            product=str(extra.get("product", "aries")),
            target_device=str(extra["target_device"]).strip() if extra.get("target_device") else None,
            model_nickname=extra.get("model_nickname"),
            optimize_option=extra.get("optimize_option"),
            singlecore_compile=extra.get("singlecore_compile"),
            save_sample=extra.get("save_sample"),
        )

    def validate(self) -> "QBBuildOptions":
        self._normalize_in_place()
        if self.quantize_method not in _QUANTIZE_METHODS:
            raise ValueError(
                "QBBuildOptions.quantize_method must be one of: "
                + ", ".join(repr(m) for m in _QUANTIZE_METHODS)
            )
        if not isinstance(self.product, str) or not self.product.strip():
            raise ValueError("QBBuildOptions.product must be a non-empty string")
        if self.calib_data_path is not None and (not isinstance(self.calib_data_path, str) or not self.calib_data_path.strip()):
            raise ValueError("QBBuildOptions.calib_data_path must be a non-empty string when provided")
        if self.target_device is not None and (not isinstance(self.target_device, str) or not self.target_device.strip()):
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

    def to_legacy_extra(self) -> Dict[str, Any]:
        extra: Dict[str, Any] = {
            "quantize_method": self.quantize_method,
            "product": self.product,
        }
        if self.use_random_calib is not None:
            extra["use_random_calib"] = self.use_random_calib
        if self.calib_data_path:
            extra["calib_data_path"] = self.calib_data_path
        if self.target_device:
            extra["target_device"] = self.target_device
        if self.model_nickname is not None:
            extra["model_nickname"] = self.model_nickname
        if self.optimize_option is not None:
            extra["optimize_option"] = self.optimize_option
        if self.singlecore_compile is not None:
            extra["singlecore_compile"] = self.singlecore_compile
        if self.save_sample is not None:
            extra["save_sample"] = self.save_sample
        return extra


@dataclass(frozen=True)
class QBVisionRuntimeOptions:
    core_mode: str | None = None
    allow_dynamic_shape: bool = False

    def _normalize_in_place(self) -> None:
        if isinstance(self.core_mode, str):
            normalized = self.core_mode.strip()
            object.__setattr__(self, "core_mode", normalized or None)

    @classmethod
    def from_legacy_extra(cls, extra: Mapping[str, Any] | None) -> "QBVisionRuntimeOptions":
        extra = dict(extra or {})
        return cls(
            core_mode=str(extra["core_mode"]).strip() if extra.get("core_mode") else None,
            allow_dynamic_shape=bool(extra.get("allow_dynamic_shape", False)),
        )

    def validate(self) -> "QBVisionRuntimeOptions":
        self._normalize_in_place()
        if self.core_mode is not None and (not isinstance(self.core_mode, str) or not self.core_mode.strip()):
            raise ValueError("QBVisionRuntimeOptions.core_mode must be a non-empty string when provided")
        return self

    def to_legacy_extra(self) -> Dict[str, Any]:
        extra: Dict[str, Any] = {"allow_dynamic_shape": self.allow_dynamic_shape}
        if self.core_mode is not None:
            extra["core_mode"] = self.core_mode
        return extra


@dataclass(frozen=True)
class QBSequenceRuntimeOptions:
    core_mode: str | None = None
    allow_dynamic_shape: bool = False

    def _normalize_in_place(self) -> None:
        if isinstance(self.core_mode, str):
            normalized = self.core_mode.strip()
            object.__setattr__(self, "core_mode", normalized or None)

    @classmethod
    def from_legacy_extra(cls, extra: Mapping[str, Any] | None) -> "QBSequenceRuntimeOptions":
        extra = dict(extra or {})
        return cls(
            core_mode=str(extra["core_mode"]).strip() if extra.get("core_mode") else None,
            allow_dynamic_shape=bool(extra.get("allow_dynamic_shape", False)),
        )

    def validate(self) -> "QBSequenceRuntimeOptions":
        self._normalize_in_place()
        if self.core_mode is not None and (not isinstance(self.core_mode, str) or not self.core_mode.strip()):
            raise ValueError("QBSequenceRuntimeOptions.core_mode must be a non-empty string when provided")
        return self

    def to_legacy_extra(self) -> Dict[str, Any]:
        extra: Dict[str, Any] = {"allow_dynamic_shape": self.allow_dynamic_shape}
        if self.core_mode is not None:
            extra["core_mode"] = self.core_mode
        return extra


def resolve_qb_build_options(options: Any, extra: Mapping[str, Any] | None) -> QBBuildOptions:
    if isinstance(options, QBBuildOptions):
        return options.validate()
    if options is not None:
        raise TypeError("BuildConfig.backend_options must be a QBBuildOptions instance when provided")
    return QBBuildOptions.from_legacy_extra(extra).validate()


def resolve_qb_runtime_options(options: Any, extra: Mapping[str, Any] | None) -> QBVisionRuntimeOptions:
    if isinstance(options, QBVisionRuntimeOptions):
        return options.validate()
    if options is not None:
        raise TypeError("RuntimeConfig.backend_options must be a QBVisionRuntimeOptions instance when provided")
    return QBVisionRuntimeOptions.from_legacy_extra(extra).validate()


def resolve_qb_sequence_runtime_options(
    options: Any,
    extra: Mapping[str, Any] | None,
) -> QBSequenceRuntimeOptions:
    if isinstance(options, QBSequenceRuntimeOptions):
        return options.validate()
    if options is not None:
        raise TypeError("SequenceRuntimeConfig.backend_options must be a QBSequenceRuntimeOptions instance when provided")
    return QBSequenceRuntimeOptions.from_legacy_extra(extra).validate()

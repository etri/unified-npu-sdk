from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


_TARGET_NPUS = ("warboy", "warboy-2pe")


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


@dataclass(frozen=True)
class WarboyBuildOptions:
    target_npu: str = "warboy-2pe"
    target_ir: str = "enf"
    compiler_config: tuple[str, ...] = ()

    def normalized(self) -> "WarboyBuildOptions":
        target_npu = str(self.target_npu).strip()
        if target_npu not in _TARGET_NPUS:
            raise ValueError(
                "WarboyBuildOptions.target_npu must be one of: "
                + ", ".join(repr(t) for t in _TARGET_NPUS)
            )
        target_ir = str(self.target_ir).strip()
        if not target_ir:
            raise ValueError("WarboyBuildOptions.target_ir must be a non-empty string")
        compiler_config = tuple(str(item) for item in self.compiler_config)
        return WarboyBuildOptions(
            target_npu=target_npu,
            target_ir=target_ir,
            compiler_config=compiler_config,
        )

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
            allow_dynamic_shape=_parse_bool(
                self.allow_dynamic_shape,
                "WarboyRuntimeOptions.allow_dynamic_shape",
            ),
        )

    def to_metadata(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {
            "device": normalized.device,
            "allow_dynamic_shape": normalized.allow_dynamic_shape,
        }


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

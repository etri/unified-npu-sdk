from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal


def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer, got {value!r}")
    return value


def _positive_int_or_none(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, field_name)


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
            tensor_parallel_size=_positive_int(
                self.tensor_parallel_size,
                "RNGDBuildOptions.tensor_parallel_size",
            ),
            pipeline_parallel_size=_positive_int(
                self.pipeline_parallel_size,
                "RNGDBuildOptions.pipeline_parallel_size",
            ),
            max_model_len=_positive_int_or_none(
                self.max_model_len,
                "RNGDBuildOptions.max_model_len",
            ),
            dry_run=bool(self.dry_run),
            optim_level=_normalize_optional_str(self.optim_level),
            build_report=bool(self.build_report),
            concurrency=_positive_int_or_none(
                self.concurrency,
                "RNGDBuildOptions.concurrency",
            ),
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
        return {
            "fxb_path": normalized.fxb_path,
            "devices": normalized.devices,
        }


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

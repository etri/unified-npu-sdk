from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal


def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _positive_int_or_none(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer when provided, got {value!r}")
    return value


@dataclass
class RNGDBuildOptions:
    build_mode: Literal["fetch", "fxb_build"] = "fetch"
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
            dry_run=bool(self.dry_run),
            optim_level=_normalize_optional_str(self.optim_level),
            build_report=bool(self.build_report),
            concurrency=_positive_int_or_none(self.concurrency, "RNGDBuildOptions.concurrency"),
        )

    @classmethod
    def from_raw(
        cls,
        raw: "RNGDBuildOptions | Dict[str, Any] | None" = None,
        legacy_extra: Dict[str, Any] | None = None,
    ) -> "RNGDBuildOptions":
        source: Dict[str, Any]
        if isinstance(raw, cls):
            return raw.normalized()
        source = dict(legacy_extra or {})
        if raw is not None:
            source.update(dict(raw))
        return cls(
            build_mode=source.get("build_mode", "fetch"),
            dry_run=bool(source.get("dry_run", False)),
            optim_level=source.get("optim_level"),
            build_report=bool(source.get("build_report", False)),
            concurrency=source.get("concurrency"),
        ).normalized()

    def to_metadata(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {
            "build_mode": normalized.build_mode,
            "dry_run": normalized.dry_run,
            "optim_level": normalized.optim_level,
            "build_report": normalized.build_report,
            "concurrency": normalized.concurrency,
        }


@dataclass
class RNGDRuntimeOptions:
    fxb_path: str | Path | None = None
    devices: str | None = None

    def normalized(self) -> "RNGDRuntimeOptions":
        fxb_path = self.fxb_path
        if isinstance(fxb_path, Path):
            fxb_path = str(fxb_path)
        fxb_path = _normalize_optional_str(fxb_path)
        return RNGDRuntimeOptions(
            fxb_path=fxb_path,
            devices=_normalize_optional_str(self.devices),
        )

    @classmethod
    def from_raw(
        cls,
        raw: "RNGDRuntimeOptions | Dict[str, Any] | None" = None,
        *,
        legacy_fxb_path: str | Path | None = None,
        legacy_devices: str | None = None,
    ) -> "RNGDRuntimeOptions":
        if isinstance(raw, cls):
            options = raw
        else:
            source = dict(raw or {})
            options = cls(
                fxb_path=source.get("fxb_path", legacy_fxb_path),
                devices=source.get("devices", legacy_devices),
            )
        return options.normalized()

    def to_metadata(self) -> Dict[str, Any]:
        normalized = self.normalized()
        return {
            "fxb_path": normalized.fxb_path,
            "devices": normalized.devices,
        }

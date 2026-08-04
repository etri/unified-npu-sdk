from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import numpy as np


def require_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def validate_shape(shape: Tuple[int, ...], field_name: str) -> Tuple[int, ...]:
    if not isinstance(shape, tuple) or not shape:
        raise ValueError(f"{field_name} must be a non-empty tuple of positive integers")
    if not all(isinstance(dim, int) and dim > 0 for dim in shape):
        raise ValueError(f"{field_name} must contain only positive integers: {shape!r}")
    return shape


def parse_bool(value: Any, field_name: str) -> bool:
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


def to_numpy(output: Any) -> Any:
    if isinstance(output, np.ndarray):
        return output
    if hasattr(output, "detach") and callable(output.detach):
        return output.detach().cpu().numpy()
    if isinstance(output, (list, tuple)):
        if len(output) == 1:
            return to_numpy(output[0])
        return [to_numpy(item) for item in output]
    return np.asarray(output)


def parse_non_negative_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return parsed


_CORE_MODE_SETTERS = {
    "auto": "set_auto_core_mode",
    "single": "set_single_core_mode",
    "global4": "set_global4_core_mode",
    "global8": "set_global8_core_mode",
    "multi": "set_multi_core_mode",
}


def build_model_config(qb_type: Any, core_mode: Optional[str]):
    ModelConfig = getattr(qb_type, "ModelConfig", None)
    if ModelConfig is None:
        return None
    mc = ModelConfig()

    if not core_mode:
        return mc

    key = core_mode.strip().lower()
    setter_name = _CORE_MODE_SETTERS.get(key)
    if setter_name is None:
        raise ValueError(
            f"Unsupported core_mode {core_mode!r}. "
            f"Use one of: {', '.join(_CORE_MODE_SETTERS)}."
        )
    fn = getattr(mc, setter_name, None)
    if not callable(fn):
        raise RuntimeError(
            f"qbruntime ModelConfig has no {setter_name}() for core_mode={core_mode!r}"
        )
    if fn() is False:
        raise RuntimeError(
            f"ModelConfig.{setter_name}() failed to set core_mode={core_mode!r}"
        )
    return mc


def load_qbruntime_modules() -> tuple[Any, Any, Any]:
    try:
        import qbruntime
        from qbruntime import model as qb_model
        from qbruntime import type as qb_type
    except Exception as exc:  # pragma: no cover - vendor SDK required
        raise RuntimeError(
            "qbruntime (QB-RUNTIME) is required to run .mxq inference. "
            "Install the Mobilint qbruntime package first (see docs.mobilint.com)."
        ) from exc
    return qbruntime, qb_model, qb_type


def validate_mxq_path(engine_path: Any) -> Path:
    p = Path(engine_path)
    if not p.exists():
        raise FileNotFoundError(f"QB model not found: {p}")
    if p.suffix != ".mxq":
        raise ValueError(f"Expected a .mxq model file, got: {p}")
    return p

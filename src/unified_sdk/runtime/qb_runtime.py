from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple
import numpy as np

from unified_sdk.runtime.registry import register
from unified_sdk.types import RuntimeConfig, RuntimeHandle


_CAPABILITY_FAMILY = "vision.direct-python-runtime"
_RUNTIME_PIPELINE = (
    "validate_runtime_config",
    "resolve_model_config",
    "load_vendor_model",
    "validate_input",
    "run_vendor_inference",
    "normalize_output",
    "destroy_runtime",
)
_VENDOR_API_MAP = {
    "model_config": "qbruntime.type.ModelConfig / qbruntime.type.CoreMode",
    "create_runtime": "qbruntime.model.load(str(mxq_path), model_config)",
    "infer": "model.infer([input_array])",
    "destroy": "model.dispose/release/unload/close best-effort",
}


def _require_non_empty_string(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"RuntimeConfig.{field_name} must be a non-empty string")
    return value.strip()


def _validate_shape(shape: Tuple[int, ...], field_name: str) -> Tuple[int, ...]:
    if not isinstance(shape, tuple) or not shape:
        raise ValueError(f"RuntimeConfig.{field_name} must be a non-empty tuple of positive integers")
    if not all(isinstance(dim, int) and dim > 0 for dim in shape):
        raise ValueError(f"RuntimeConfig.{field_name} must contain only positive integers: {shape!r}")
    return shape


def _parse_device(value: Any) -> int:
    try:
        device = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"RuntimeConfig.extra['device'] must be an integer, got {value!r}") from exc
    if device < 0:
        raise ValueError("RuntimeConfig.extra['device'] must be >= 0")
    return device


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
    raise ValueError(f"RuntimeConfig.extra['{field_name}'] must be a boolean-like value, got {value!r}")


def _to_numpy(output: Any) -> np.ndarray:
    if isinstance(output, np.ndarray):
        return output
    if hasattr(output, "detach") and callable(output.detach):
        return output.detach().cpu().numpy()
    if isinstance(output, (list, tuple)):
        if len(output) != 1:
            raise TypeError(
                "QB runtime returned multiple outputs; pass a single-output model "
                "or handle the raw list output directly"
            )
        return _to_numpy(output[0])
    return np.asarray(output)


def _build_model_config(qb_type, core_mode: Optional[str]):
    """qbruntime.type.ModelConfig 를 best-effort 로 구성한다.

    core_mode 예: 'global8' (참조 테스트 가이드 기준). qbruntime.type.CoreMode 는
    문서상 enum 이며 버전에 따라 멤버/세터 이름이 다를 수 있어, 안전하게 매칭하고
    실패하면 기본 ModelConfig (None) 로 위임한다.
    """
    ModelConfig = getattr(qb_type, "ModelConfig", None)
    if ModelConfig is None:
        return None
    try:
        mc = ModelConfig()
    except Exception:
        return None

    if core_mode:
        CoreMode = getattr(qb_type, "CoreMode", None)
        member = None
        if CoreMode is not None:
            member = (
                getattr(CoreMode, core_mode, None)
                or getattr(CoreMode, core_mode.upper(), None)
                or getattr(CoreMode, core_mode.capitalize(), None)
            )
        if member is not None:
            for setter in ("set_core_mode", "set_global_core_mode", "set_single_core_mode"):
                fn = getattr(mc, setter, None)
                if callable(fn):
                    try:
                        fn(member)
                        break
                    except Exception:
                        continue
    return mc


class _QBRuntime:
    """Mobilint ARISE(QB) runtime adapter — wraps qbruntime (QB-RUNTIME).

    문서 근거 API (docs.mobilint.com/v1.2, group PythonAPI):
      - qbruntime.model.load(path, model_config=None) -> Model  (NPU 업로드까지 단일 스텝)
      - qbruntime.type.ModelConfig / CoreMode          (코어 모드/할당)
      - qbruntime.accelerator.Accelerator              (NPU)
      - qbruntime.type.get_available_device_numbers()
    """

    name = "qb"

    def create(self, cfg: RuntimeConfig) -> RuntimeHandle:
        if cfg.backend != self.name:
            raise ValueError(f"QB runtime adapter received backend={cfg.backend!r}")

        p = Path(cfg.engine_path)
        if not p.exists():
            raise FileNotFoundError(f"QB model not found: {p}")
        if p.suffix != ".mxq":
            raise ValueError(f"Expected a .mxq model file, got: {p}")

        input_name = _require_non_empty_string(cfg.input_name, "input_name")
        output_name = _require_non_empty_string(cfg.output_name, "output_name")
        input_shape = _validate_shape(tuple(cfg.input_shape), "input_shape")

        extra = dict(cfg.extra or {})
        device = _parse_device(extra.get("device", 0))
        core_mode = extra.get("core_mode")  # 예: 'global8'
        if core_mode is not None and (not isinstance(core_mode, str) or not core_mode.strip()):
            raise ValueError("RuntimeConfig.extra['core_mode'] must be a non-empty string when provided")

        try:
            import qbruntime
            from qbruntime import model as qb_model
            from qbruntime import type as qb_type
        except Exception as exc:  # pragma: no cover - 벤더 SDK 필요
            raise RuntimeError(
                "qbruntime (QB-RUNTIME) is required to run .mxq inference. "
                "Install the Mobilint qbruntime package first (see docs.mobilint.com)."
            ) from exc

        model_config = _build_model_config(qb_type, core_mode)
        try:
            model = qb_model.load(str(p), model_config)
        except Exception as exc:
            raise RuntimeError(f"Failed to load QB model for {p}: {exc}") from exc

        return RuntimeHandle(
            backend=self.name,
            engine_path=str(p),
            input_name=input_name,
            output_name=output_name,
            input_shape=input_shape,
            ctx={
                "model": model,
                "device": device,
                "core_mode": core_mode,
                "extra": extra,
                "capability_family": _CAPABILITY_FAMILY,
                "runtime_pipeline": _RUNTIME_PIPELINE,
                "vendor_api_map": _VENDOR_API_MAP,
            },
        )

    def infer(self, rh: RuntimeHandle, input_array: np.ndarray) -> np.ndarray:
        if not rh.ctx or "model" not in rh.ctx:
            raise RuntimeError("QB RuntimeHandle is closed or invalid")

        model = rh.ctx["model"]
        extra = rh.ctx.get("extra", {})
        allow_dynamic = _parse_bool(extra.get("allow_dynamic_shape", False), "allow_dynamic_shape")

        input_shape = tuple(getattr(input_array, "shape", ()))
        if (not allow_dynamic) and input_shape != tuple(rh.input_shape):
            raise ValueError(
                f"Bad input shape: {getattr(input_array, 'shape', None)}, expected {rh.input_shape}"
            )

        # qbruntime Model.infer 는 list[np.ndarray] 입력 -> list[np.ndarray] 출력.
        try:
            out = model.infer([input_array])
        except Exception as exc:
            raise RuntimeError(f"QB inference failed: {exc}") from exc

        return _to_numpy(out)

    def destroy(self, rh: RuntimeHandle) -> None:
        model = rh.ctx.get("model") if rh.ctx else None
        if model is not None:
            for method in ("dispose", "release", "unload", "close"):
                fn = getattr(model, method, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
                    break
        rh.ctx.clear()


register(_QBRuntime())

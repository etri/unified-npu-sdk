from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple
import numpy as np

from unified_sdk.runtime.registry import register
from unified_sdk.types import BatchParam, RuntimeConfig, RuntimeHandle


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
    "infer": "model.infer([input_array], cache_size=..., params=...)",
    "destroy": "model.dispose/release/unload/close best-effort",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "qbruntime.type.ModelConfig / CoreMode": "RuntimeConfig.extra['core_mode']",
    "qbruntime.model.load(str(mxq_path), model_config)": "create_runtime(cfg)",
    "model.infer([input_array], cache_size=..., params=...)": "infer(rh, input_array, cache_size=..., batch_params=...)",
    "qbruntime output tensor/list": "infer(...) return np.ndarray or list[np.ndarray]",
    "model.dispose/release/unload/close": "destroy_runtime(rh)",
}


def describe_api_mapping() -> dict[str, Any]:
    return {
        "unified_api": {
            "create": "create_runtime(cfg)",
            "infer": "infer(rh, input_array, cache_size=..., batch_params=...)",
            "destroy": "destroy_runtime(rh)",
        },
        "backend": "qb",
        "capability_family": _CAPABILITY_FAMILY,
        "mapping_direction": "vendor_api ==> unified_api",
        "pipeline": _RUNTIME_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "vendor_to_unified_api_map": _VENDOR_TO_UNIFIED_API_MAP,
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


def _to_numpy(output: Any) -> Any:
    if isinstance(output, np.ndarray):
        return output
    if hasattr(output, "detach") and callable(output.detach):
        return output.detach().cpu().numpy()
    if isinstance(output, (list, tuple)):
        if len(output) == 1:
            return _to_numpy(output[0])
        return [_to_numpy(item) for item in output]
    return np.asarray(output)


def _parse_non_negative_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return parsed


def _normalize_batch_params(batch_params: Optional[Sequence[Any]], qbruntime_module: Any) -> Optional[list[Any]]:
    if batch_params is None:
        return None

    BatchParamCls = getattr(qbruntime_module, "BatchParam", None)
    if BatchParamCls is None:
        raise RuntimeError("qbruntime.BatchParam is unavailable but batch_params were provided")

    normalized = []
    for idx, item in enumerate(batch_params):
        if isinstance(item, BatchParamCls):
            normalized.append(item)
            continue

        if isinstance(item, BatchParam):
            sequence_length = item.sequence_length
            cache_size = item.cache_size
            cache_id = item.cache_id
        elif isinstance(item, dict):
            sequence_length = item.get("sequence_length")
            cache_size = item.get("cache_size", 0)
            cache_id = item.get("cache_id", idx)
        else:
            sequence_length = getattr(item, "sequence_length", None)
            cache_size = getattr(item, "cache_size", 0)
            cache_id = getattr(item, "cache_id", idx)

        sequence_length = _parse_non_negative_int(sequence_length, f"batch_params[{idx}].sequence_length")
        cache_size = _parse_non_negative_int(cache_size, f"batch_params[{idx}].cache_size")
        cache_id = _parse_non_negative_int(cache_id, f"batch_params[{idx}].cache_id")
        if sequence_length <= 0:
            raise ValueError(f"batch_params[{idx}].sequence_length must be > 0")

        normalized.append(BatchParamCls(sequence_length, cache_size, cache_id))
    return normalized


# core_mode 문자열 -> qbruntime.type.ModelConfig 의 전용 세터 이름.
# ModelConfig 는 CoreMode 인자를 받는 범용 세터가 아니라 모드별 세터를 제공한다
# (set_global8_core_mode() 등, 인자 없이 호출). mxq 가 컴파일된 모드와 반드시
# 일치해야 하며, 불일치 시 로드가 Model_MXQAndModelConfigNotMatch 로 실패한다.
_CORE_MODE_SETTERS = {
    "auto": "set_auto_core_mode",
    "single": "set_single_core_mode",
    "global4": "set_global4_core_mode",
    "global8": "set_global8_core_mode",
    "multi": "set_multi_core_mode",
}


def _build_model_config(qb_type, core_mode: Optional[str]):
    """qbruntime.type.ModelConfig 를 구성한다.

    core_mode 예: 'global8'. 지정된 모드를 반드시 반영하며, 반영에 실패하면
    (기본 Auto 로 지정되지 않고) 명시적으로 예외를 던진다. 기본 Auto 는
    mxq 가 단일 코어 모드일 때만 유효하기 때문이다.
    """
    ModelConfig = getattr(qb_type, "ModelConfig", None)
    if ModelConfig is None:
        return None
    mc = ModelConfig()

    if not core_mode:
        return mc  # 기본 auto-core mode (단일 코어 모드 mxq 에서만 유효)

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
                "qbruntime": qbruntime,
                "device": device,
                "core_mode": core_mode,
                "extra": extra,
                "capability_family": _CAPABILITY_FAMILY,
                "runtime_pipeline": _RUNTIME_PIPELINE,
                "vendor_api_map": _VENDOR_API_MAP,
            },
        )

    def infer(
        self,
        rh: RuntimeHandle,
        input_array: np.ndarray,
        *,
        cache_size: int = 0,
        batch_params: Optional[Sequence[BatchParam]] = None,
    ) -> Any:
        if not rh.ctx or "model" not in rh.ctx:
            raise RuntimeError("QB RuntimeHandle is closed or invalid")

        model = rh.ctx["model"]
        qbruntime_module = rh.ctx.get("qbruntime")
        extra = rh.ctx.get("extra", {})
        allow_dynamic = _parse_bool(extra.get("allow_dynamic_shape", False), "allow_dynamic_shape")
        cache_size = _parse_non_negative_int(cache_size, "cache_size")
        normalized_batch_params = _normalize_batch_params(batch_params, qbruntime_module)

        input_shape = tuple(getattr(input_array, "shape", ()))
        if (not allow_dynamic) and input_shape != tuple(rh.input_shape):
            raise ValueError(
                f"Bad input shape: {getattr(input_array, 'shape', None)}, expected {rh.input_shape}"
            )

        # qbruntime Model.infer 는 list[np.ndarray] 입력 -> list[np.ndarray] 출력.
        try:
            if normalized_batch_params is not None:
                out = model.infer([input_array], params=normalized_batch_params)
            elif cache_size > 0:
                out = model.infer([input_array], cache_size=cache_size)
            else:
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

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple
import numpy as np

from unified_sdk.options import resolve_warboy_runtime_options
from unified_sdk.runtime.registry import register
from unified_sdk.types import InferOutput, RuntimeConfig, RuntimeHandle


_CAPABILITY_FAMILY = "vision.cli-compiled-runtime"
_RUNTIME_PIPELINE = (
    "validate_runtime_config",
    "load_vendor_runner",
    "validate_input",
    "run_vendor_inference",
    "normalize_output",
    "destroy_runtime",
)
_VENDOR_API_MAP = {
    "create_runtime": "furiosa.runtime.sync.create_runner(str(enf_path), device=...)",
    "infer": "runner.run([input_array])",
    "destroy": "runner.close() or runner.__exit__(...) best-effort",
}
_VENDOR_TO_UNIFIED_API_MAP = {
    "furiosa.runtime.sync.create_runner(str(enf_path), device=...)": "create_runtime(cfg)",
    "runner.run([input_array])": "infer(rh, input_array)",
    "runner output tensor/list": "infer(...) return np.ndarray | list[np.ndarray]",
    "runner.close() / runner.__exit__(...)": "destroy_runtime(rh)",
}


def describe_api_mapping() -> dict[str, Any]:
    return {
        "unified_api": {
            "create": "create_runtime(cfg)",
            "infer": "infer(rh, input_array)",
            "destroy": "destroy_runtime(rh)",
        },
        "backend": "warboy",
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


def _one_to_numpy(value: Any) -> np.ndarray:
    # furiosa 출력 텐서는 .numpy() 를 제공하거나 numpy 로 변환 가능하다.
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "numpy") and callable(value.numpy):
        return value.numpy()
    if hasattr(value, "detach") and callable(value.detach):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_numpy(output: Any) -> InferOutput:
    if isinstance(output, (list, tuple)):
        if len(output) == 1:
            return _one_to_numpy(output[0])
        return [_one_to_numpy(item) for item in output]
    return _one_to_numpy(output)


class _WarboyRuntime:
    """FuriosaAI Warboy runtime adapter — wraps furiosa.runtime (sync).

    참조 API (furiosa-sdk 0.10.2, developer.furiosa.ai / 사내 추론 가이드 9.3B):
      - from furiosa.runtime import sync
      - runner = sync.create_runner(str(enf))     # ENF 로드
      - outputs = runner.run(inputs)              # inputs: list[np.ndarray]
      - runner.close()
    """

    name = "warboy"

    def create(self, cfg: RuntimeConfig) -> RuntimeHandle:
        if cfg.backend != self.name:
            raise ValueError(f"Warboy runtime adapter received backend={cfg.backend!r}")

        p = Path(cfg.engine_path)
        if not p.exists():
            raise FileNotFoundError(f"Warboy model not found: {p}")
        if p.suffix != ".enf":
            raise ValueError(f"Expected a .enf model file, got: {p}")

        input_name = _require_non_empty_string(cfg.input_name, "input_name")
        output_name = _require_non_empty_string(cfg.output_name, "output_name") if cfg.output_name is not None else None
        input_shape = _validate_shape(tuple(cfg.input_shape), "input_shape")
        options = resolve_warboy_runtime_options(cfg.backend_options)
        device = options.device  # 예: "warboy(0)*2" 또는 None (기본)

        try:
            from furiosa.runtime import sync
        except Exception as exc:  # pragma: no cover - 벤더 SDK 필요
            raise RuntimeError(
                "furiosa.runtime is required to run .enf inference. "
                "Install furiosa-sdk first (see developer.furiosa.ai)."
            ) from exc

        try:
            if device is not None:
                runner = sync.create_runner(str(p), device=str(device))
            else:
                runner = sync.create_runner(str(p))
        except TypeError:
            # device kwarg 미지원 버전 대비
            runner = sync.create_runner(str(p))
        except Exception as exc:
            raise RuntimeError(f"Failed to create Warboy runner for {p}: {exc}") from exc

        return RuntimeHandle(
            backend=self.name,
            engine_path=str(p),
            input_name=input_name,
            output_name=output_name,
            input_shape=input_shape,
            ctx={
                "runner": runner,
                "device": device,
                "backend_options": options.to_metadata(),
                "capability_family": _CAPABILITY_FAMILY,
                "runtime_pipeline": _RUNTIME_PIPELINE,
                "vendor_api_map": _VENDOR_API_MAP,
            },
        )

    def infer(self, rh: RuntimeHandle, input_array: np.ndarray) -> InferOutput:
        if not rh.ctx or "runner" not in rh.ctx:
            raise RuntimeError("Warboy RuntimeHandle is closed or invalid")

        runner = rh.ctx["runner"]
        options = rh.ctx.get("backend_options", {})
        allow_dynamic = bool(options.get("allow_dynamic_shape", False))

        input_shape = tuple(getattr(input_array, "shape", ()))
        if (not allow_dynamic) and input_shape != tuple(rh.input_shape):
            raise ValueError(
                f"Bad input shape: {getattr(input_array, 'shape', None)}, expected {rh.input_shape}"
            )

        # furiosa runner.run 은 list[np.ndarray] 입력을 받는다. 일부 버전은 단일 배열도 허용.
        try:
            out = runner.run([input_array])
        except TypeError:
            out = runner.run(input_array)
        except Exception as exc:
            raise RuntimeError(f"Warboy inference failed: {exc}") from exc

        return _to_numpy(out)

    def destroy(self, rh: RuntimeHandle) -> None:
        runner = rh.ctx.get("runner") if rh.ctx else None
        if runner is not None:
            for method in ("close", "__exit__"):
                fn = getattr(runner, method, None)
                if callable(fn):
                    try:
                        fn() if method == "close" else fn(None, None, None)
                    except Exception:
                        pass
                    break
        rh.ctx.clear()


register(_WarboyRuntime())

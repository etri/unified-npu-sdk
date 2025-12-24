from pathlib import Path
from typing import Any, Dict
import numpy as np

from unified_sdk.runtime.registry import register
from unified_sdk.types import RuntimeConfig, RuntimeHandle

class _RBLNRuntime:
    name = "rbln"

    def create(self, cfg: RuntimeConfig) -> RuntimeHandle:
        import rebel

        p = Path(cfg.engine_path)
        if not p.exists():
            raise FileNotFoundError(f"RBLN model not found: {p}")

        extra = cfg.extra or {}
        device = int(extra.get("device", 0))
        tensor_type = str(extra.get("tensor_type", "np"))  # 'np'|'pt'

        runtime = rebel.Runtime(
            str(p),
            device=device,
            tensor_type=tensor_type,
            activate_profiler=bool(extra.get("activate_profiler", False)),
            timeout=extra.get("timeout", None),
        )

        return RuntimeHandle(
            backend=self.name,
            engine_path=str(p),
            input_name=cfg.input_name,
            output_name=cfg.output_name,
            input_shape=cfg.input_shape,
            ctx={"runtime": runtime, "tensor_type": tensor_type, "extra": extra},
        )

    def infer(self, rh: RuntimeHandle, input_array: np.ndarray) -> np.ndarray:
        rt = rh.ctx["runtime"]
        extra = rh.ctx.get("extra", {})
        allow_dynamic = bool(extra.get("allow_dynamic_shape", False))

        # bucketing 쓸 거면 shape 고정 검증은 옵션으로 빼는 게 안전
        if (not allow_dynamic) and tuple(getattr(input_array, "shape", ())) != tuple(rh.input_shape):
            raise ValueError(f"Bad input shape: {getattr(input_array,'shape',None)}, expected {rh.input_shape}")

        out = rt(input_array)  # Runtime.__call__은 run과 동일

        # tensor_type='pt'면 torch.Tensor가 나올 수 있음
        if rh.ctx["tensor_type"] == "pt":
            return out.detach().cpu().numpy()
        return out

    def destroy(self, rh: RuntimeHandle) -> None:
        rh.ctx.clear()

register(_RBLNRuntime())


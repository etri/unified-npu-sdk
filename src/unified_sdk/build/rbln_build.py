from pathlib import Path
from typing import Any, Dict

from unified_sdk.build.registry import register
from unified_sdk.types import BuildConfig, BuildResult


class _RBLNBuildAdapter:
    name = "rbln"

    def build(self, cfg: BuildConfig) -> BuildResult:
        import torch
        import rebel

        model = cfg.model_or_path
        if not hasattr(model, "eval"):
            raise TypeError(
                "For rbln backend, BuildConfig.model_or_path must be a torch.nn.Module"
            )
        model.eval()

        dtype = torch.float16 if cfg.precision == "fp16" else torch.float32
        name = cfg.input_name or "input"
        extra = cfg.extra or {}
        npu = extra.get("npu")

        if cfg.bucketing_shapes:
            # bucketing: 여러 입력 shape를 리스트의 리스트 형태로
            input_info = [
                [(name, list(shape), dtype)] for shape in cfg.bucketing_shapes
            ]
        else:
            input_info = [(name, list(cfg.input_shape), dtype)]

        compile_kwargs: Dict[str, Any] = {"input_info": input_info}
        if npu:
            compile_kwargs["npu"] = npu

        compiled = rebel.compile_from_torch(model, **compile_kwargs)

        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rbln_path = out_dir / f"{cfg.model_name}.rbln"
        compiled.save(str(rbln_path))

        meta: Dict[str, Any] = {
            "backend": self.name,
            "rbln_path": str(rbln_path),
            "input_info": input_info,
            "npu": npu,
            "precision": cfg.precision,
            "extra": extra,
        }
        return BuildResult(
            backend=self.name,
            compiled_model_path=str(rbln_path),
            meta_data=meta,
        )


register(_RBLNBuildAdapter())

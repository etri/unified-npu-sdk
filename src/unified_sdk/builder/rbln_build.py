from pathlib import Path
from typing import Any, Dict, List, Tuple

from unified_sdk.build.registry import register
from unified_sdk.types import BuildConfig, BuildResult

class _RBLNBuildAdapter:
    name = "rbln"

    def build(self, cfg: BuildConfig) -> BuildResult:
        import torch
        import rebel

        # 1) 모델 확보 (여기서 torch.nn.Module이어야 함)
        model = cfg.model_or_path
        if not hasattr(model, "eval"):
            raise TypeError("For rbln backend, BuildConfig.model_or_path must be a torch.nn.Module")

        model.eval()

        # 2) dtype 매핑 (compile_from_torch는 dtype에 torch.dtype도 받음)
        
        dtype = torch.float16 if cfg.precision == "fp16" else torch.float32

        # 3) input_info 구성: (name, shape(list[int]), dtype) - 기본은 opt_input_shape 사용
        name = cfg.input_name or "input"
        s_min, s_opt, s_max = list(cfg.min_input_shape), list(cfg.opt_input_shape), list(cfg.max_input_shape)

        use_bucketing = bool((cfg.extra or {}).get("bucketing", False))
        if use_bucketing:
            # bucketing: input_info를 "여러 입력 설정 리스트의 리스트"로 넣을 수 있음
            input_info = [
                [(name, s_min, dtype)],
                [(name, s_opt, dtype)],
                [(name, s_max, dtype)],
            ]
        else:
            input_info = [(name, s_opt, dtype)]

        compiled = rebel.compile_from_torch(model, input_info=input_info) 

        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rbln_path = out_dir / f"{cfg.model_name}.rbln"
        compiled.save(str(rbln_path))

        meta: Dict[str, Any] = {
            "backend": self.name,
            "rbln_path": str(rbln_path),
            "input_info": input_info,
            "precision": cfg.precision,
            "extra": cfg.extra or {},
        }
        return BuildResult(backend=self.name, compiled_model_path=str(rbln_path), meta_data=meta)

register(_RBLNBuildAdapter())


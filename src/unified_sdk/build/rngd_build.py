from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from unified_sdk.build.registry import register
from unified_sdk.types import BuildConfig, BuildResult


_CAPABILITY_FAMILY = "llm.artifact-and-generation"
_BUILD_PIPELINE = (
    "validate_config",
    "resolve_model_ref_or_artifact",
    "resolve_parallel_and_model_options",
    "run_artifact_builder_or_return_model_ref",
    "verify_artifact",
    "emit_metadata",
)
_VENDOR_API_MAP = {
    "provided_model_ref": "HF model id or existing artifact directory",
    "compile": "furiosa_llm.ArtifactBuilder(...).build(str(out_dir))",
    "parallel_config": "furiosa_llm.ParallelConfig(tensor_parallel_size=..., pipeline_parallel_size=...)",
    "model_config": "furiosa_llm.ModelConfig(max_model_len=...)",
    "artifact": "artifact directory or HF model id",
}


def _require_positive_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"BuildConfig.{field_name} must be a positive integer, got {value!r}")
    return value


def _capability_metadata(extra: Dict[str, Any], source: str) -> Dict[str, Any]:
    return {
        "capability_family": _CAPABILITY_FAMILY,
        "build_pipeline": _BUILD_PIPELINE,
        "vendor_api_map": _VENDOR_API_MAP,
        "selected_path": source,
        "compile_options": {
            "compile": bool(extra.get("compile", False)),
            "bucket_config": extra.get("bucket_config"),
        },
    }


class _RNGDBuildAdapter:
    """FuriosaAI RNGD build adapter — wraps furiosa-llm ArtifactBuilder.

    LLM 스택이라 vision 백엔드의 '컴파일'과 의미가 다르다. 두 경로를 지원한다
    (fetch 기본 + compile 훅):
      1) fetch (기본): HF 모델 id (예: 'furiosa-ai/Qwen2.5-0.5B-Instruct') 또는
         기존 아티팩트 디렉터리를 그대로 사용한다. LLM 이 로드 시 처리한다.
      2) compile 훅 (extra['compile']=True): ArtifactBuilder 로 AOT 컴파일하여
         아티팩트 디렉터리를 생성한다.

    참조: developer.furiosa.ai (furiosa_llm.ArtifactBuilder / model-preparation).
    """

    name = "rngd"

    def build(self, cfg: BuildConfig) -> BuildResult:
        if cfg.backend != self.name:
            raise ValueError(f"RNGD build adapter received backend={cfg.backend!r}")

        extra = dict(cfg.extra or {})
        model_ref = str(cfg.model_or_path)
        compile_flag = bool(extra.get("compile", False))

        # ---- Path 1: fetch / provided (HF 모델 id 또는 기존 아티팩트 dir) ----
        if not compile_flag:
            meta: Dict[str, Any] = {
                "backend": self.name,
                "source": "provided",
                "model_ref": model_ref,
                "note": "HF model id or existing artifact dir; loaded by furiosa_llm.LLM at runtime",
                "extra": extra,
                **_capability_metadata(extra, "model_ref"),
            }
            return BuildResult(
                backend=self.name,
                compiled_model_path=model_ref,
                meta_data=meta,
            )

        # ---- Path 2: ArtifactBuilder AOT 컴파일 (compile hook) ----
        tp = _require_positive_int(cfg.tensor_parallel_size, "tensor_parallel_size")
        pp = _require_positive_int(cfg.pipeline_parallel_size, "pipeline_parallel_size")
        out_dir = Path(cfg.out_dir) / cfg.model_name
        out_dir.parent.mkdir(parents=True, exist_ok=True)

        try:
            from furiosa_llm import ArtifactBuilder
        except Exception as exc:  # pragma: no cover - 벤더 SDK 필요
            raise RuntimeError(
                "furiosa-llm is required to build an RNGD artifact. "
                "Install furiosa-llm first (see developer.furiosa.ai)."
            ) from exc

        builder_kwargs: Dict[str, Any] = {}
        try:
            from furiosa_llm import ParallelConfig

            builder_kwargs["parallel_config"] = ParallelConfig(
                tensor_parallel_size=tp,
                pipeline_parallel_size=pp,
            )
        except Exception:
            pass
        if cfg.max_model_len:
            try:
                from furiosa_llm import ModelConfig

                builder_kwargs["model_config"] = ModelConfig(max_model_len=int(cfg.max_model_len))
            except Exception:
                pass

        try:
            builder = ArtifactBuilder(model_id_or_path=model_ref, **builder_kwargs)
        except TypeError:
            # 인자 이름이 다른 버전 대비 (첫 positional 로 시도)
            builder = ArtifactBuilder(model_ref, **builder_kwargs)

        try:
            builder.build(str(out_dir))
        except Exception as exc:
            raise RuntimeError(f"furiosa-llm ArtifactBuilder.build failed: {exc}") from exc

        if not out_dir.is_dir():
            raise RuntimeError(
                f"ArtifactBuilder reported success but artifact dir not found at {out_dir}"
            )

        meta = {
            "backend": self.name,
            "source": "artifact_builder",
            "artifact_dir": str(out_dir),
            "model_ref": model_ref,
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp,
            "max_model_len": cfg.max_model_len,
            "extra": extra,
            **_capability_metadata(extra, "artifact_builder"),
        }
        return BuildResult(
            backend=self.name,
            compiled_model_path=str(out_dir),
            meta_data=meta,
        )


register(_RNGDBuildAdapter())

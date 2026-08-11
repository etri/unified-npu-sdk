from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_sdk.frontends import (  # noqa: E402
    classify_tensorrt_llm_source,
    resolve_tensorrt_llm_fetch_request,
    resolve_tensorrt_llm_build_request,
    resolve_tensorrt_vision_build_request,
)
from unified_sdk.frontends.types import (  # noqa: E402
    TensorRTLLMFrontendFetchRequest,
    TensorRTLLMFrontendBuildRequest,
    TensorRTVisionFrontendBuildRequest,
)


class TensorRTFrontendTests(unittest.TestCase):
    def test_resolve_provided_engine_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models = root / "models"
            out = root / "build_output"
            models.mkdir()
            out.mkdir()
            engine = models / "demo.engine"
            engine.write_bytes(b"engine-bytes")
            resolved = resolve_tensorrt_vision_build_request(
                TensorRTVisionFrontendBuildRequest(
                    model_name="demo",
                    models_dir=models,
                    out_dir=out,
                    provided_engine=engine,
                )
            )
            self.assertEqual(resolved.kind, "provided_artifact")
            self.assertEqual(resolved.prepared_input.kind, "provided_artifact")
            self.assertEqual(resolved.prepared_input.provided_artifact.source_path, engine.resolve())

    def test_resolve_explicit_onnx_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models = root / "models"
            out = root / "build_output"
            models.mkdir()
            out.mkdir()
            onnx = models / "demo.onnx"
            onnx.write_bytes(b"onnx")
            resolved = resolve_tensorrt_vision_build_request(
                TensorRTVisionFrontendBuildRequest(
                    model_name="demo",
                    models_dir=models,
                    out_dir=out,
                    onnx_path=onnx,
                    input_name="images",
                    min_input_shape=(1, 3, 640, 640),
                    opt_input_shape=(1, 3, 640, 640),
                    max_input_shape=(1, 3, 640, 640),
                )
            )
            self.assertEqual(resolved.kind, "onnx_path")
            self.assertEqual(resolved.prepared_input.kind, "compile_source")
            self.assertEqual(resolved.prepared_input.compile_source.input_name, "images")

    def test_resolve_provided_engine_request_tracks_requested_precision_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models = root / "models"
            out = root / "build_output"
            models.mkdir()
            out.mkdir()
            engine = models / "demo.engine"
            engine.write_bytes(b"engine-bytes")
            resolved = resolve_tensorrt_vision_build_request(
                TensorRTVisionFrontendBuildRequest(
                    model_name="demo",
                    models_dir=models,
                    out_dir=out,
                    precision="fp16",
                    provided_engine=engine,
                )
            )
            self.assertEqual(
                resolved.prepared_input.provided_artifact.destination_path,
                (out / "demo_FP16.engine").resolve(),
            )

    def test_resolve_llm_fetch_request(self) -> None:
        resolved = resolve_tensorrt_llm_fetch_request(
            TensorRTLLMFrontendFetchRequest(
                model_ref="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            )
        )
        self.assertEqual(resolved.kind, "runtime_model_ref")
        self.assertEqual(resolved.prepared_input.kind, "runtime_model_ref")
        self.assertEqual(resolved.prepared_input.source_kind, "model_id")

    def test_resolve_llm_local_artifact_fetch_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "tinyllama_trtllm"
            artifact_dir.mkdir()
            (artifact_dir / "rank0.engine").write_bytes(b"engine")
            (artifact_dir / "config.json").write_text("{}")
            resolved = resolve_tensorrt_llm_fetch_request(
                TensorRTLLMFrontendFetchRequest(
                    model_ref=artifact_dir,
                )
            )
            self.assertEqual(resolved.kind, "runtime_model_ref")
            self.assertEqual(resolved.prepared_input.source_kind, "local_artifact_dir")

    def test_resolve_llm_plain_local_hf_dir_fetch_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "my-finetuned-qwen"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}")
            (model_dir / "tokenizer_config.json").write_text("{}")
            (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")
            resolved = resolve_tensorrt_llm_fetch_request(
                TensorRTLLMFrontendFetchRequest(
                    model_ref=model_dir,
                )
            )
            self.assertEqual(resolved.kind, "runtime_model_ref")
            self.assertEqual(resolved.prepared_input.source_kind, "local_model_path")

    def test_classify_tensorrt_llm_source_distinguishes_hf_and_artifact_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            hf_dir = root / "hf-model"
            hf_dir.mkdir()
            (hf_dir / "config.json").write_text("{}")
            (hf_dir / "model.safetensors").write_bytes(b"weights")

            artifact_dir = root / "artifact-dir"
            artifact_dir.mkdir()
            (artifact_dir / "executor_config.json").write_text("{}")

            self.assertEqual(classify_tensorrt_llm_source(hf_dir)[0], "local_model_path")
            self.assertEqual(classify_tensorrt_llm_source(artifact_dir)[0], "local_artifact_dir")

    def test_resolve_llm_artifact_build_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            resolved = resolve_tensorrt_llm_build_request(
                TensorRTLLMFrontendBuildRequest(
                    model_ref="meta-llama/Llama-3.2-1B-Instruct",
                    out_dir=out_dir,
                    model_name="llama",
                )
            )
            self.assertEqual(resolved.kind, "artifact_build")
            self.assertEqual(resolved.prepared_input.artifact_dir, (out_dir / "llama").resolve())
            self.assertEqual(resolved.prepared_input.compile_variant, "model_ref_api")

    def test_resolve_llm_checkpoint_compile_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "config.json").write_text("{}")
            (checkpoint_dir / "rank0.safetensors").write_bytes(b"weights")
            resolved = resolve_tensorrt_llm_build_request(
                TensorRTLLMFrontendBuildRequest(
                    model_ref=checkpoint_dir,
                    out_dir=Path(tmpdir) / "artifacts",
                    model_name="llama",
                )
            )
            self.assertEqual(resolved.kind, "artifact_build")
            self.assertEqual(resolved.prepared_input.source_kind, "local_checkpoint_dir")
            self.assertEqual(resolved.prepared_input.compile_variant, "checkpoint_dir_cli")

    def test_resolve_llm_checkpoint_fetch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "config.json").write_text("{}")
            (checkpoint_dir / "rank0.safetensors").write_bytes(b"weights")
            with self.assertRaises(ValueError):
                resolve_tensorrt_llm_fetch_request(
                    TensorRTLLMFrontendFetchRequest(
                        model_ref=checkpoint_dir,
                    )
                )

    def test_resolve_llm_artifact_compile_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir) / "tinyllama_trtllm"
            artifact_dir.mkdir()
            (artifact_dir / "rank0.engine").write_bytes(b"engine")
            (artifact_dir / "config.json").write_text("{}")
            with self.assertRaises(ValueError):
                resolve_tensorrt_llm_build_request(
                    TensorRTLLMFrontendBuildRequest(
                        model_ref=artifact_dir,
                        out_dir=Path(tmpdir) / "artifacts",
                        model_name="tinyllama",
                    )
                )


if __name__ == "__main__":
    unittest.main()

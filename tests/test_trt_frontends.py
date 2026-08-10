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
    resolve_tensorrt_llm_build_request,
    resolve_tensorrt_vision_build_request,
)
from unified_sdk.frontends.types import (  # noqa: E402
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

    def test_resolve_llm_fetch_request(self) -> None:
        resolved = resolve_tensorrt_llm_build_request(
            TensorRTLLMFrontendBuildRequest(
                model_ref="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                out_dir=Path("artifacts"),
                model_name="tinyllama",
                build_mode="fetch",
            )
        )
        self.assertEqual(resolved.kind, "runtime_model_ref")
        self.assertEqual(resolved.prepared_input.kind, "runtime_model_ref")

    def test_resolve_llm_artifact_build_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            resolved = resolve_tensorrt_llm_build_request(
                TensorRTLLMFrontendBuildRequest(
                    model_ref="meta-llama/Llama-3.2-1B-Instruct",
                    out_dir=out_dir,
                    model_name="llama",
                    build_mode="llm_api_compile",
                )
            )
            self.assertEqual(resolved.kind, "artifact_build")
            self.assertEqual(resolved.prepared_input.artifact_dir, (out_dir / "llama").resolve())


if __name__ == "__main__":
    unittest.main()

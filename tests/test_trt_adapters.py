from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_sdk.build.api import build_unified, build_unified_LLM  # noqa: E402
from unified_sdk.frontends import prepare_tensorrt_vision_build_input  # noqa: E402
from unified_sdk.frontends.types import PreparedTensorRTLLMBuildInput  # noqa: E402
from unified_sdk.options import TensorRTLLMBuildOptions, TensorRTVisionBuildOptions  # noqa: E402
from unified_sdk.types import BuildConfig, LLMBuildConfig  # noqa: E402
from unified_sdk.runtime.tensorrt_runtime import _TensorRTRuntime  # noqa: E402
from unified_sdk.types import RuntimeHandle  # noqa: E402


class TensorRTAdapterTests(unittest.TestCase):
    def test_vision_build_copies_provided_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            src_engine = root / "models" / "demo.engine"
            src_engine.parent.mkdir(parents=True)
            src_engine.write_bytes(b"engine")
            out_dir = root / "build_output"
            prepared = prepare_tensorrt_vision_build_input(
                src_engine,
                out_dir / "demo_FP16.engine",
                source_label="provided_engine",
                provenance_kind="provided_artifact",
                provenance_detail="provided engine",
                input_name="input",
                min_input_shape=(1, 3, 224, 224),
                opt_input_shape=(1, 3, 224, 224),
                max_input_shape=(1, 3, 224, 224),
            )
            result = build_unified(
                BuildConfig(
                    backend="tensorrt",
                    model_or_path=str(src_engine),
                    out_dir=out_dir,
                    model_name="demo",
                    backend_options=TensorRTVisionBuildOptions(precision="fp16"),
                    prepared_input=prepared,
                )
            )
            self.assertTrue(Path(result.compiled_model_path).is_file())
            self.assertEqual(Path(result.compiled_model_path).read_bytes(), b"engine")
            self.assertEqual(result.meta_data["prepared_kind"], "provided_artifact")
            self.assertTrue(result.compiled_model_path.endswith("demo_FP16.engine"))

    def test_vision_build_rejects_compile_without_prepared_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            onnx = root / "demo.onnx"
            onnx.write_bytes(b"onnx")
            with self.assertRaises(ValueError):
                build_unified(
                    BuildConfig(
                        backend="tensorrt",
                        model_or_path=str(onnx),
                        out_dir=root / "build_output",
                        model_name="demo",
                        backend_options=TensorRTVisionBuildOptions(),
                    )
                )

    def test_llm_build_fetch_passthrough(self) -> None:
        result = build_unified_LLM(
            LLMBuildConfig(
                backend="tensorrt",
                model_or_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                out_dir="artifacts",
                model_name="tinyllama",
                backend_options=TensorRTLLMBuildOptions(build_mode="fetch", tensor_parallel_size=2, max_model_len=1024),
                prepared_input=PreparedTensorRTLLMBuildInput(
                    kind="runtime_model_ref",
                    model_ref="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    artifact_dir=None,
                ),
            )
        )
        self.assertEqual(result.compiled_model_path, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.assertEqual(result.meta_data["prepared_kind"], "runtime_model_ref")
        self.assertEqual(result.meta_data["backend_options"]["tensor_parallel_size"], 2)

    def test_llm_artifact_build_requires_prepared_input(self) -> None:
        with self.assertRaises(ValueError):
            build_unified_LLM(
                LLMBuildConfig(
                    backend="tensorrt",
                    model_or_path="repo/model",
                    out_dir="artifacts",
                    model_name="demo",
                    backend_options=TensorRTLLMBuildOptions(build_mode="llm_api_compile"),
                )
            )

    def test_runtime_dynamic_shape_option_is_explicitly_not_implemented_for_rebind(self) -> None:
        rh = RuntimeHandle(
            backend="tensorrt",
            engine_path="demo.engine",
            input_name="input",
            output_name="output",
            input_shape=(1, 3, 224, 224),
            ctx={
                "context": object(),
                "cuda": object(),
                "allow_dynamic_shape": True,
                "h_input": None,
                "h_output": None,
            },
        )
        with self.assertRaises(NotImplementedError):
            _TensorRTRuntime().infer(rh, __import__("numpy").zeros((1, 3, 256, 256), dtype="float32"))


if __name__ == "__main__":
    unittest.main()

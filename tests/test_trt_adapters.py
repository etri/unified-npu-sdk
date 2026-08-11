from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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
                    source_kind="model_id",
                    artifact_dir=None,
                ),
            )
        )
        self.assertEqual(result.compiled_model_path, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.assertEqual(result.meta_data["prepared_kind"], "runtime_model_ref")
        self.assertEqual(result.meta_data["resolved_phase"], "fetch_contract_only")
        self.assertFalse(result.meta_data["artifact_emitted"])
        self.assertTrue(result.meta_data["runtime_may_trigger_vendor_build"])
        self.assertEqual(result.meta_data["backend_options"]["tensor_parallel_size"], 2)

    def test_llm_artifact_build_requires_prepared_input(self) -> None:
        with self.assertRaises(ValueError):
            build_unified_LLM(
                LLMBuildConfig(
                    backend="tensorrt",
                    model_or_path="repo/model",
                    out_dir="artifacts",
                    model_name="demo",
                    backend_options=TensorRTLLMBuildOptions(build_mode="custom_compile"),
                )
            )

    def test_llm_checkpoint_compile_invokes_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_dir = root / "checkpoint"
            checkpoint_dir.mkdir()
            artifact_dir = root / "artifacts" / "llama"
            with patch("unified_sdk.build.tensorrt_llm_build.subprocess.run") as patched:
                result = build_unified_LLM(
                    LLMBuildConfig(
                        backend="tensorrt",
                        model_or_path=str(checkpoint_dir),
                        out_dir=root / "artifacts",
                        model_name="llama",
                        backend_options=TensorRTLLMBuildOptions(build_mode="custom_compile", max_model_len=2048),
                        prepared_input=PreparedTensorRTLLMBuildInput(
                            kind="artifact_build",
                            model_ref=str(checkpoint_dir),
                            source_kind="local_checkpoint_dir",
                            source_path=checkpoint_dir,
                            artifact_dir=artifact_dir,
                            compile_variant="checkpoint_dir_cli",
                            checkpoint_dir=checkpoint_dir,
                        ),
                    )
                )
            patched.assert_called_once()
            self.assertEqual(result.compiled_model_path, str(artifact_dir.resolve()))
            self.assertEqual(result.meta_data["compile_variant"], "checkpoint_dir_cli")
            self.assertEqual(result.meta_data["resolved_phase"], "custom_compile_artifact")
            self.assertTrue(result.meta_data["artifact_emitted"])
            self.assertFalse(result.meta_data["runtime_may_trigger_vendor_build"])

    def test_llm_prepared_fetch_rejects_custom_compile_mode(self) -> None:
        with self.assertRaises(ValueError):
            build_unified_LLM(
                LLMBuildConfig(
                    backend="tensorrt",
                    model_or_path="repo/model",
                    out_dir="artifacts",
                    model_name="demo",
                    backend_options=TensorRTLLMBuildOptions(build_mode="custom_compile"),
                    prepared_input=PreparedTensorRTLLMBuildInput(
                        kind="runtime_model_ref",
                        model_ref="repo/model",
                        source_kind="model_id",
                    ),
                )
            )

    def test_llm_checkpoint_cli_rejects_nonauthoritative_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_dir = root / "checkpoint"
            checkpoint_dir.mkdir()
            artifact_dir = root / "artifacts" / "llama"
            with self.assertRaises(ValueError):
                build_unified_LLM(
                    LLMBuildConfig(
                        backend="tensorrt",
                        model_or_path=str(checkpoint_dir),
                        out_dir=root / "artifacts",
                        model_name="llama",
                        backend_options=TensorRTLLMBuildOptions(
                            build_mode="custom_compile",
                            tensor_parallel_size=2,
                            dtype="float16",
                        ),
                        prepared_input=PreparedTensorRTLLMBuildInput(
                            kind="artifact_build",
                            model_ref=str(checkpoint_dir),
                            source_kind="local_checkpoint_dir",
                            source_path=checkpoint_dir,
                            artifact_dir=artifact_dir,
                            compile_variant="checkpoint_dir_cli",
                            checkpoint_dir=checkpoint_dir,
                        ),
                    )
                )

    def test_runtime_dynamic_shape_option_rebinds_buffers(self) -> None:
        class FakeDeviceBuffer:
            def __init__(self, size: int):
                self.size = size
                self.freed = False

            def __int__(self):
                return self.size

            def free(self):
                self.freed = True

        class FakeStream:
            handle = 123

            def synchronize(self):
                return None

        class FakeCuda:
            def pagelocked_empty(self, size, dtype):
                return __import__("numpy").zeros(size, dtype=dtype)

            def mem_alloc(self, nbytes):
                return FakeDeviceBuffer(nbytes)

            def memcpy_htod_async(self, dst, src, stream):
                return None

            def memcpy_dtoh_async(self, dst, src, stream):
                dst[...] = 0

        class FakeContext:
            def __init__(self):
                self.current_shape = (1, 3, 224, 224)
                self.bound = {}

            def set_input_shape(self, name, shape):
                self.current_shape = tuple(shape)

            def get_tensor_shape(self, name):
                if name == "output":
                    return (self.current_shape[0], 1000)
                return self.current_shape

            def set_tensor_address(self, name, addr):
                self.bound[name] = addr

            def execute_async_v3(self, stream_handle):
                return True

        class FakeEngine:
            num_bindings = 2

            def get_binding_index(self, name):
                return 0 if name == "input" else 1

        rh = RuntimeHandle(
            backend="tensorrt",
            engine_path="demo.engine",
            input_name="input",
            output_name="output",
            input_shape=(1, 3, 224, 224),
            ctx={
                "context": FakeContext(),
                "engine": FakeEngine(),
                "cuda": FakeCuda(),
                "allow_dynamic_shape": True,
                "h_input": __import__("numpy").zeros((1, 3, 224, 224), dtype="float32"),
                "h_output": __import__("numpy").zeros((1, 1000), dtype="float32"),
                "d_input": FakeDeviceBuffer(1),
                "d_output": FakeDeviceBuffer(2),
                "stream": FakeStream(),
                "in_dtype": __import__("numpy").float32,
                "out_dtype": __import__("numpy").float32,
                "use_v3": True,
            },
        )
        out = _TensorRTRuntime().infer(rh, __import__("numpy").zeros((1, 3, 256, 256), dtype="float32"))
        self.assertEqual(rh.input_shape, (1, 3, 256, 256))
        self.assertEqual(tuple(rh.ctx["h_input"].shape), (1, 3, 256, 256))
        self.assertEqual(tuple(rh.ctx["h_output"].shape), (1, 1000))
        self.assertEqual(tuple(out.shape), (1, 1000))


if __name__ == "__main__":
    unittest.main()

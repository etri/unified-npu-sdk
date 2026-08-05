from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_sdk.build.rbln_build import _RBLNBuildAdapter  # noqa: E402
from unified_sdk.build.rbln_llm_build import build_llm  # noqa: E402
from unified_sdk.frontends.types import (  # noqa: E402
    PreparedRBLNLLMBuildInput,
    PreparedRBLNCompileSource,
    PreparedRBLNVisionBuildInput,
    ProvidedRBLNArtifact,
)
from unified_sdk.options import (  # noqa: E402
    RBLNLLMBuildOptions,
    RBLNLLMRuntimeOptions,
    RBLNVisionBuildOptions,
    RBLNVisionRuntimeOptions,
)
from unified_sdk.runtime.rbln_llm_runtime import create_llm, destroy_llm, generate_llm  # noqa: E402
from unified_sdk.runtime.rbln_runtime import _RBLNRuntime  # noqa: E402
from unified_sdk.types import BuildConfig, LLMBuildConfig, LLMRuntimeConfig, RuntimeConfig  # noqa: E402


class _FakeRunner:
    def __call__(self, input_array):
        return np.asarray(input_array)


class RBLNAdaptersTest(unittest.TestCase):
    def test_build_adapter_places_provided_rbln(self) -> None:
        adapter = _RBLNBuildAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src = tmp_path / "source.rbln"
            src.write_text("rbln")
            cfg = BuildConfig(
                backend="rbln",
                model_or_path=str(src),
                out_dir=str(tmp_path / "builds"),
                model_name="resnet50",
                backend_options=RBLNVisionBuildOptions(),
            )
            result = adapter.build(cfg)
            self.assertTrue(Path(result.compiled_model_path).is_file())
            self.assertEqual(result.meta_data["source"], "provided")

    def test_build_adapter_prefers_prepared_input_contract(self) -> None:
        adapter = _RBLNBuildAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src = tmp_path / "prepared.rbln"
            src.write_text("rbln")
            cfg = BuildConfig(
                backend="rbln",
                model_or_path="should_not_be_used",
                out_dir=str(tmp_path / "builds"),
                model_name="resnet50",
                backend_options=RBLNVisionBuildOptions(),
                prepared_input=PreparedRBLNVisionBuildInput(
                    kind="provided_artifact",
                    provided_artifact=ProvidedRBLNArtifact(
                        source_path=src,
                        destination_path=tmp_path / "builds" / "resnet50.rbln",
                    ),
                ),
            )
            result = adapter.build(cfg)
            self.assertEqual(result.meta_data["origin"], str(src))

    def test_build_adapter_requires_prepared_input_for_compile_source(self) -> None:
        adapter = _RBLNBuildAdapter()
        cfg = BuildConfig(
            backend="rbln",
            model_or_path="microsoft/resnet-50",
            out_dir="builds",
            model_name="resnet50",
            backend_options=RBLNVisionBuildOptions(compile_frontend="optimum_image_classification"),
        )
        with self.assertRaises(RuntimeError):
            adapter.build(cfg)

    def test_runtime_adapter_uses_backend_options(self) -> None:
        adapter = _RBLNRuntime()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            rbln_path = tmp_path / "resnet50.rbln"
            rbln_path.write_text("rbln")
            fake_runtime = types.SimpleNamespace(Runtime=mock.Mock(return_value=_FakeRunner()))

            cfg = RuntimeConfig(
                backend="rbln",
                engine_path=str(rbln_path),
                input_name="input",
                output_name="output",
                input_shape=(1, 3, 224, 224),
                backend_options=RBLNVisionRuntimeOptions(device=1, tensor_type="np"),
            )
            with mock.patch.dict(sys.modules, {"rebel": fake_runtime}):
                rh = adapter.create(cfg)
                out = adapter.infer(rh, np.zeros((1, 3, 224, 224), dtype=np.float32))
                adapter.destroy(rh)
            fake_runtime.Runtime.assert_called_once()
            self.assertIsInstance(out, np.ndarray)

    def test_llm_build_fetch_mode_uses_backend_options(self) -> None:
        result = build_llm(
            LLMBuildConfig(
                backend="rbln",
                model_or_path="Qwen/Qwen3-0.6B",
                backend_options=RBLNLLMBuildOptions(build_mode="fetch"),
                prepared_input=PreparedRBLNLLMBuildInput(
                    kind="runtime_model_ref",
                    model_ref="Qwen/Qwen3-0.6B",
                ),
            )
        )
        self.assertEqual(result.compiled_model_path, "Qwen/Qwen3-0.6B")
        self.assertEqual(result.meta_data["build_mode"], "fetch")
        self.assertFalse(result.meta_data["artifact_emitted"])

    def test_llm_runtime_uses_backend_options(self) -> None:
        fake_outputs = [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="hello")])]
        fake_llm = types.SimpleNamespace(generate=mock.Mock(return_value=fake_outputs), llm_engine=None, engine=None)
        fake_vllm = types.SimpleNamespace(LLM=mock.Mock(return_value=fake_llm), SamplingParams=lambda **params: params)

        cfg = LLMRuntimeConfig(
            backend="rbln",
            engine_path="Qwen/Qwen3-0.6B",
            backend_options=RBLNLLMRuntimeOptions(runtime_impl="vllm", block_size=256, dtype="float16"),
        )
        with mock.patch.dict(sys.modules, {"vllm": fake_vllm}):
            rh = create_llm(cfg)
            text = generate_llm(rh, "hello")
            destroy_llm(rh)
        fake_vllm.LLM.assert_called_once()
        self.assertEqual(text, "hello")


if __name__ == "__main__":
    unittest.main()

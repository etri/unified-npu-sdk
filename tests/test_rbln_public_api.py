from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_sdk.build.api import build_unified, build_unified_LLM  # noqa: E402
from unified_sdk.options import RBLNLLMBuildOptions, RBLNVisionRuntimeOptions  # noqa: E402
from unified_sdk.runtime.api import create_runtime, destroy_runtime, generate_LLM, infer  # noqa: E402
from unified_sdk.types import BuildConfig, LLMBuildConfig, RuntimeConfig  # noqa: E402


class RBLNPublicApiTest(unittest.TestCase):
    def test_build_unified_dispatches_to_registered_builder(self) -> None:
        fake_builder = mock.Mock()
        fake_builder.build.return_value = "ok"
        cfg = BuildConfig(backend="rbln", model_or_path="models/resnet50.rbln")
        with mock.patch("unified_sdk.build.api.get_builder", return_value=fake_builder):
            result = build_unified(cfg)
        self.assertEqual(result, "ok")

    def test_create_runtime_infer_destroy_dispatch(self) -> None:
        fake_adapter = mock.Mock()
        fake_handle = mock.Mock(backend="rbln")
        fake_adapter.create.return_value = fake_handle
        fake_adapter.infer.return_value = np.zeros((1, 1), dtype=np.float32)
        cfg = RuntimeConfig(
            backend="rbln",
            engine_path="builds/resnet50.rbln",
            input_name="input",
            output_name="output",
            input_shape=(1, 3, 224, 224),
            backend_options=RBLNVisionRuntimeOptions(),
        )
        with mock.patch("unified_sdk.runtime.api.get_runtime", return_value=fake_adapter):
            rh = create_runtime(cfg)
            out = infer(rh, np.zeros((1, 3, 224, 224), dtype=np.float32))
            destroy_runtime(rh)
        self.assertIs(rh, fake_handle)
        self.assertEqual(out.shape, (1, 1))

    def test_build_unified_llm_calls_llm_builder(self) -> None:
        cfg = LLMBuildConfig(
            backend="rbln",
            model_or_path="Qwen/Qwen3-0.6B",
            backend_options=RBLNLLMBuildOptions(build_mode="fetch"),
        )
        with mock.patch("unified_sdk.build.api._rbln_llm.build_llm", return_value="llm-ok"):
            result = build_unified_LLM(cfg)
        self.assertEqual(result, "llm-ok")

    def test_generate_llm_surface_delegates(self) -> None:
        fake_handle = mock.Mock()
        with mock.patch("unified_sdk.runtime.api._rbln_llm.generate_llm", return_value="hello"):
            text = generate_LLM(fake_handle, "prompt")
        self.assertEqual(text, "hello")


if __name__ == "__main__":
    unittest.main()

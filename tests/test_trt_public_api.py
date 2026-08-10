from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_sdk.build import api as build_api  # noqa: E402
from unified_sdk.runtime import api as runtime_api  # noqa: E402
from unified_sdk.types import BuildResult, LLMBuildConfig, LLMRuntimeConfig, LLMRuntimeHandle  # noqa: E402


class TensorRTPublicApiTests(unittest.TestCase):
    def test_build_unified_llm_uses_registry_dispatch(self) -> None:
        class FakeAdapter:
            def build(self, cfg):
                return BuildResult(backend="tensorrt", compiled_model_path="artifact", meta_data={"path": "ok"})

        with patch("unified_sdk.build.api.get_llm_builder", return_value=FakeAdapter()) as patched:
            result = build_api.build_unified_LLM(
                LLMBuildConfig(backend="tensorrt", model_or_path="repo/model", model_name="demo")
            )
        patched.assert_called_once_with("tensorrt")
        self.assertEqual(result.compiled_model_path, "artifact")

    def test_runtime_llm_uses_registry_dispatch(self) -> None:
        class FakeAdapter:
            def create(self, cfg):
                return LLMRuntimeHandle(backend="tensorrt", engine_path=str(cfg.engine_path), ctx={"ok": True})

            def infer(self, rh, prompt, **overrides):
                return f"{prompt}:{overrides.get('max_tokens', 'none')}"

            def destroy(self, rh):
                rh.ctx["destroyed"] = True

        with patch("unified_sdk.runtime.api.get_llm_runtime", return_value=FakeAdapter()) as patched:
            rh = runtime_api.create_runtime_LLM(LLMRuntimeConfig(backend="tensorrt", engine_path="repo/model"))
            text = runtime_api.generate_LLM(rh, "hello", max_tokens=3)
            runtime_api.destroy_runtime_LLM(rh)
        self.assertEqual(patched.call_count, 3)
        self.assertEqual(text, "hello:3")
        self.assertTrue(rh.ctx["destroyed"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_sdk.build.rngd_build import _RNGDBuildAdapter
from unified_sdk.options import RNGDBuildOptions, RNGDRuntimeOptions
from unified_sdk.runtime.rngd_runtime import _RNGDRuntime
from unified_sdk.types import LLMBuildConfig, LLMRuntimeConfig


class _FakeSamplingParams:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeOutputText:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeRequestOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [_FakeOutputText(text)]


class _FakeLLM:
    def __init__(self, engine: str, **kwargs) -> None:
        self.engine = engine
        self.kwargs = kwargs
        self.closed = False

    def generate(self, prompts, sampling):
        return [_FakeRequestOutput(f"reply:{prompt}:{sampling.kwargs['max_tokens']}") for prompt in prompts]

    def close(self) -> None:
        self.closed = True


class RNGDAdapterTests(unittest.TestCase):
    def test_build_adapter_fetch_returns_model_ref(self) -> None:
        adapter = _RNGDBuildAdapter()
        cfg = LLMBuildConfig(
            model_or_path="furiosa-ai/Qwen2.5-0.5B-Instruct",
            backend_options=RNGDBuildOptions(build_mode="fetch"),
        )
        result = adapter.build(cfg)
        self.assertEqual(result.compiled_model_path, "furiosa-ai/Qwen2.5-0.5B-Instruct")
        self.assertEqual(result.meta_data["source"], "provided")

    def test_build_adapter_runs_fxb_build_with_mocked_subprocess(self) -> None:
        adapter = _RNGDBuildAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            model_dir.mkdir()
            cfg = LLMBuildConfig(
                model_or_path=str(model_dir),
                out_dir=root / "artifacts",
                model_name="demo",
                backend_options=RNGDBuildOptions(
                    build_mode="fxb_build",
                    tensor_parallel_size=8,
                    pipeline_parallel_size=1,
                ),
            )

            def _fake_run(cmd, check=False, capture_output=True, text=True):
                Path(cmd[3]).write_bytes(b"fxb")
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

            with patch("unified_sdk.build.rngd_build.subprocess.run", side_effect=_fake_run):
                result = adapter.build(cfg)

            self.assertTrue(Path(result.compiled_model_path).is_file())
            self.assertEqual(result.meta_data["source"], "fxb_build")
            self.assertEqual(result.meta_data["tensor_parallel_size"], 8)

    def test_runtime_adapter_uses_typed_runtime_options(self) -> None:
        adapter = _RNGDRuntime()
        fake_module = SimpleNamespace(LLM=_FakeLLM, SamplingParams=_FakeSamplingParams)
        cfg = LLMRuntimeConfig(
            engine_path="furiosa-ai/Qwen2.5-0.5B-Instruct",
            max_tokens=64,
            backend_options=RNGDRuntimeOptions(fxb_path="/tmp/demo.fxb", devices="npu:0"),
        )
        with patch.dict("sys.modules", {"furiosa_llm": fake_module}):
            rh = adapter.create(cfg)
            text = adapter.generate(rh, "hello")
            adapter.destroy(rh)
        self.assertEqual(text, "reply:hello:64")
        self.assertEqual(rh.ctx, {})


if __name__ == "__main__":
    unittest.main()

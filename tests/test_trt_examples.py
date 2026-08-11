from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _load_wrapper_module():
    script_path = ROOT / "examples" / "run_tensorrt_llm_prepare_checkpoint.py"
    spec = importlib.util.spec_from_file_location("trt_llm_prepare_wrapper", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules.setdefault("trt_llm_prepare_wrapper", module)
    spec.loader.exec_module(module)
    return module


class TensorRTExampleWrapperTests(unittest.TestCase):
    def test_prepare_helper_finds_qwen_vendor_script_from_env(self) -> None:
        module = _load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            vendor_root = Path(tmpdir) / "TensorRT-LLM"
            script = vendor_root / "examples" / "models" / "core" / "qwen" / "convert_checkpoint.py"
            script.parent.mkdir(parents=True)
            script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            with patch.dict(os.environ, {"TENSORRT_LLM_SRC": str(vendor_root)}, clear=False):
                found = module._find_vendor_convert_script(ROOT, "qwen")
        self.assertEqual(found, script.resolve())

    def test_prepare_helper_finds_legacy_llama_vendor_script(self) -> None:
        module = _load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            vendor_root = Path(tmpdir) / "TensorRT-LLM"
            script = vendor_root / "examples" / "llama" / "convert_checkpoint.py"
            script.parent.mkdir(parents=True)
            script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            with patch.dict(os.environ, {"TENSORRT_LLM_SRC": str(vendor_root)}, clear=False):
                found = module._find_vendor_convert_script(ROOT, "llama")
        self.assertEqual(found, script.resolve())

    def test_prepare_checkpoint_helper_forwards_cli_and_env(self) -> None:
        module = _load_wrapper_module()
        captured: dict[str, object] = {}
        fake_vendor_script = Path("/tmp/TensorRT-LLM/examples/models/core/qwen/convert_checkpoint.py")

        def _fake_run(cmd, env=None):
            captured["cmd"] = cmd
            captured["env"] = env
            return types.SimpleNamespace(returncode=0)

        with patch.object(module, "_find_vendor_convert_script", return_value=fake_vendor_script), \
             patch.object(module.subprocess, "run", side_effect=_fake_run):
            rc = module.main(
                [
                    "--model-ref",
                    "./models/Qwen2.5-0.5B-Instruct",
                    "--output-dir",
                    "./models/qwen25_trtllm_ckpt",
                    "--dtype",
                    "float16",
                    "--model-family",
                    "qwen",
                    "--tensorrt-llm-src",
                    "../TensorRT-LLM",
                    "--load-by-shard",
                ]
            )

        self.assertEqual(rc, 0)
        cmd = captured["cmd"]
        env = captured["env"]
        self.assertIsInstance(cmd, list)
        self.assertIn("--model_dir", cmd)
        self.assertIn("--output_dir", cmd)
        self.assertIn("--load_by_shard", cmd)
        self.assertEqual(str(fake_vendor_script), cmd[1])
        self.assertEqual(
            env["TENSORRT_LLM_SRC"],
            str(Path("../TensorRT-LLM").resolve()),
        )


if __name__ == "__main__":
    unittest.main()

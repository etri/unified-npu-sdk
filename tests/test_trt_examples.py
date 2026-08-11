from __future__ import annotations

import importlib.util
import types
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _load_wrapper_module():
    script_path = ROOT / "examples" / "llama" / "convert_checkpoint.py"
    spec = importlib.util.spec_from_file_location("trt_llama_convert_wrapper", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules.setdefault("trt_llama_convert_wrapper", module)
    spec.loader.exec_module(module)
    return module


class TensorRTExampleWrapperTests(unittest.TestCase):
    def test_convert_checkpoint_wrapper_finds_vendor_script_from_env(self) -> None:
        module = _load_wrapper_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            vendor_root = Path(tmpdir) / "TensorRT-LLM"
            script = vendor_root / "examples" / "llama" / "convert_checkpoint.py"
            script.parent.mkdir(parents=True)
            script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            with patch.dict(os.environ, {"TENSORRT_LLM_SRC": str(vendor_root)}, clear=False):
                found = module._find_vendor_convert_script(ROOT)
        self.assertEqual(found, script.resolve())

    def test_convert_checkpoint_wrapper_uses_installed_api_when_available(self) -> None:
        module = _load_wrapper_module()

        class FakeModel:
            saved = None

            @classmethod
            def from_hugging_face(cls, model_dir, dtype="float16", mapping=None, load_by_shard=False):
                inst = cls()
                inst.model_dir = model_dir
                inst.dtype = dtype
                inst.mapping = mapping
                inst.load_by_shard = load_by_shard
                return inst

            def save_checkpoint(self, output_dir, save_config=True):
                self.saved = (output_dir, save_config)

        fake_models = types.SimpleNamespace(LLaMAForCausalLM=FakeModel)
        fake_tllm = types.SimpleNamespace(models=fake_models)

        with patch.dict(sys.modules, {"tensorrt_llm": fake_tllm, "tensorrt_llm.models": fake_models}, clear=False):
            ok = module._convert_with_installed_api(
                types.SimpleNamespace(
                    model_dir="./models/TinyLlama-1.1B-Chat-v1.0",
                    output_dir="./models/tinyllama_trtllm_ckpt",
                    dtype="float16",
                    tp_size=None,
                    pp_size=None,
                    workers=None,
                    load_by_shard=False,
                )
            )
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()

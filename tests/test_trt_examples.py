from __future__ import annotations

import importlib.util
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


if __name__ == "__main__":
    unittest.main()

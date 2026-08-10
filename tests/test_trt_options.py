from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from unified_sdk.options import (  # noqa: E402
    TensorRTLLMBuildOptions,
    TensorRTVisionBuildOptions,
    resolve_tensorrt_llm_runtime_options,
    resolve_tensorrt_vision_runtime_options,
)


class TensorRTOptionTests(unittest.TestCase):
    def test_vision_build_options_normalize_precision(self) -> None:
        options = TensorRTVisionBuildOptions(precision="FP16", workspace_mib=512).normalized()
        self.assertEqual(options.precision, "fp16")
        self.assertEqual(options.workspace_mib, 512)

    def test_vision_runtime_legacy_extra_rejects_unknown_keys(self) -> None:
        with self.assertRaises(ValueError):
            resolve_tensorrt_vision_runtime_options(None, extra={"bad_key": True})

    def test_llm_runtime_options_legacy_extra_is_typed(self) -> None:
        options = resolve_tensorrt_llm_runtime_options(
            None,
            extra={
                "tokenizer_path": "tok-dir",
                "tensor_parallel_size": 2,
                "max_model_len": 2048,
                "dtype": "float16",
                "trust_remote_code": True,
            },
        )
        self.assertEqual(options.tensor_parallel_size, 2)
        self.assertEqual(options.max_model_len, 2048)
        self.assertEqual(options.dtype, "float16")
        self.assertTrue(options.trust_remote_code)

    def test_llm_build_options_reject_unknown_legacy_extra(self) -> None:
        with self.assertRaises(ValueError):
            TensorRTLLMBuildOptions.from_legacy_extra({"build_mode": "fetch", "foo": "bar"})


if __name__ == "__main__":
    unittest.main()

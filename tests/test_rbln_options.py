from __future__ import annotations

import sys
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_sdk.options import (  # noqa: E402
    RBLNLLMBuildOptions,
    RBLNLLMRuntimeOptions,
    RBLNVisionBuildOptions,
    RBLNVisionRuntimeOptions,
    resolve_rbln_llm_build_options,
    resolve_rbln_llm_runtime_options,
    resolve_rbln_vision_build_options,
    resolve_rbln_vision_runtime_options,
)


class RBLNOptionsTest(unittest.TestCase):
    def test_resolve_vision_build_options_from_legacy_extra(self) -> None:
        options = resolve_rbln_vision_build_options(
            None,
            extra={
                "npu": "RBLN-CA22",
                "precision": "fp32",
                "model_trace_method": "jittrace",
                "compile_frontend": "rebel",
            },
        )
        self.assertEqual(options.npu, "RBLN-CA22")
        self.assertEqual(options.precision, "fp32")
        self.assertEqual(options.model_trace_method, "jittrace")

    def test_resolve_vision_runtime_options_from_instance(self) -> None:
        options = resolve_rbln_vision_runtime_options(
            RBLNVisionRuntimeOptions(device=1, tensor_type="pt", allow_dynamic_shape=True)
        )
        self.assertEqual(options.device, 1)
        self.assertEqual(options.tensor_type, "pt")
        self.assertTrue(options.allow_dynamic_shape)

    def test_resolve_llm_build_options_from_legacy_extra(self) -> None:
        options = resolve_rbln_llm_build_options(
            None,
            extra={"build_mode": "optimum_compile", "trust_remote_code": True, "revision": "main"},
        )
        self.assertEqual(options.build_mode, "optimum_compile")
        self.assertTrue(options.trust_remote_code)
        self.assertEqual(options.revision, "main")

    def test_resolve_llm_runtime_options_from_instance(self) -> None:
        options = resolve_rbln_llm_runtime_options(
            RBLNLLMRuntimeOptions(
                runtime_impl="vllm",
                tensor_parallel_size=2,
                max_model_len=1024,
                block_size=256,
                dtype="float16",
            )
        )
        self.assertEqual(options.runtime_impl, "vllm")
        self.assertEqual(options.tensor_parallel_size, 2)
        self.assertEqual(options.max_model_len, 1024)
        self.assertEqual(options.block_size, 256)
        self.assertEqual(options.dtype, "float16")


if __name__ == "__main__":
    unittest.main()

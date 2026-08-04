from __future__ import annotations

import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_sdk.options import (
    RNGDBuildOptions,
    RNGDRuntimeOptions,
    resolve_rngd_build_options,
    resolve_rngd_runtime_options,
)


class RNGDOptionsTests(unittest.TestCase):
    def test_build_options_default_resolution(self) -> None:
        options = resolve_rngd_build_options(None)
        self.assertEqual(options.build_mode, "fetch")
        self.assertEqual(options.tensor_parallel_size, 1)
        self.assertEqual(options.pipeline_parallel_size, 1)

    def test_build_options_normalizes_fields(self) -> None:
        options = resolve_rngd_build_options(
            RNGDBuildOptions(
                build_mode="fxb_build",
                tensor_parallel_size=8,
                pipeline_parallel_size=2,
                max_model_len=4096,
                dry_run=True,
                optim_level=" O2 ",
            )
        )
        self.assertEqual(options.optim_level, "O2")
        self.assertEqual(options.max_model_len, 4096)

    def test_runtime_options_default_resolution(self) -> None:
        options = resolve_rngd_runtime_options(None)
        self.assertIsNone(options.fxb_path)
        self.assertIsNone(options.devices)

    def test_runtime_options_normalizes_pathlike_fields(self) -> None:
        options = resolve_rngd_runtime_options(
            RNGDRuntimeOptions(fxb_path=Path("/tmp/demo.fxb"), devices=" npu:0 ")
        )
        self.assertEqual(options.fxb_path, "/tmp/demo.fxb")
        self.assertEqual(options.devices, "npu:0")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_sdk.options import (  # noqa: E402
    WarboyBuildOptions,
    WarboyRuntimeOptions,
    resolve_warboy_build_options,
    resolve_warboy_runtime_options,
)


class WarboyOptionsTest(unittest.TestCase):
    def test_build_options_normalize_defaults(self) -> None:
        options = resolve_warboy_build_options(None)
        self.assertEqual(options.target_npu, "warboy-2pe")
        self.assertEqual(options.target_ir, "enf")

    def test_build_options_reject_invalid_target(self) -> None:
        with self.assertRaises(ValueError):
            WarboyBuildOptions(target_npu="rngd").normalized()

    def test_runtime_options_normalize_values(self) -> None:
        options = resolve_warboy_runtime_options(
            WarboyRuntimeOptions(device="  warboy(0)*2  ", allow_dynamic_shape="true")
        )
        self.assertEqual(options.device, "warboy(0)*2")
        self.assertTrue(options.allow_dynamic_shape)


if __name__ == "__main__":
    unittest.main()

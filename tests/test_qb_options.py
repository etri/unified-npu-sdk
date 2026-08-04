from __future__ import annotations

import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_sdk.options import (
    QBBuildOptions,
    QBSequenceRuntimeOptions,
    QBVisionRuntimeOptions,
    resolve_qb_build_options,
    resolve_qb_runtime_options,
    resolve_qb_sequence_runtime_options,
)


class QBOptionsTests(unittest.TestCase):
    def test_build_options_default_resolution(self) -> None:
        options = resolve_qb_build_options(None)
        self.assertIsInstance(options, QBBuildOptions)
        self.assertEqual(options.quantize_method, "percentile")

    def test_runtime_options_default_resolution(self) -> None:
        options = resolve_qb_runtime_options(None)
        self.assertIsInstance(options, QBVisionRuntimeOptions)
        self.assertIsNone(options.core_mode)

    def test_sequence_options_default_resolution(self) -> None:
        options = resolve_qb_sequence_runtime_options(None)
        self.assertIsInstance(options, QBSequenceRuntimeOptions)
        self.assertIsNone(options.core_mode)


if __name__ == "__main__":
    unittest.main()

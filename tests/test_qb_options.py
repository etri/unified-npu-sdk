from __future__ import annotations

import unittest
import warnings
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
    def test_build_options_warn_on_legacy_extra_fallback(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            options = resolve_qb_build_options(None, {"quantize_method": "max"})
        self.assertIsInstance(options, QBBuildOptions)
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_runtime_options_warn_on_legacy_extra_fallback(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            options = resolve_qb_runtime_options(None, {"core_mode": "auto"})
        self.assertIsInstance(options, QBVisionRuntimeOptions)
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))

    def test_sequence_options_warn_on_legacy_extra_fallback(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            options = resolve_qb_sequence_runtime_options(None, {"core_mode": "global8"})
        self.assertIsInstance(options, QBSequenceRuntimeOptions)
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))


if __name__ == "__main__":
    unittest.main()

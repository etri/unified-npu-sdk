from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import unified_sdk  # noqa: E402
from unified_sdk.build.api import build_unified, describe_build_api_mapping  # noqa: E402
from unified_sdk.runtime.api import (  # noqa: E402
    create_runtime,
    describe_runtime_api_mapping,
    destroy_runtime,
    infer,
)
from unified_sdk.types import BuildConfig, RuntimeConfig, RuntimeHandle  # noqa: E402


class WarboyPublicApiTest(unittest.TestCase):
    def test_top_level_exports_include_frontend_and_options(self) -> None:
        self.assertIn("WarboyBuildOptions", unified_sdk.__all__)
        self.assertIn("WarboyRuntimeOptions", unified_sdk.__all__)
        self.assertIn("WarboyFrontendBuildRequest", unified_sdk.__all__)

    def test_build_api_dispatches_to_registry(self) -> None:
        fake_builder = types.SimpleNamespace(build=mock.Mock(return_value="built"))
        cfg = BuildConfig(
            backend="warboy",
            model_or_path="models/resnet50.enf",
            out_dir="builds",
            model_name="resnet50",
            input_name="input",
            input_shape=(1, 3, 224, 224),
        )
        with mock.patch("unified_sdk.build.api.get_builder", return_value=fake_builder):
            result = build_unified(cfg)
        fake_builder.build.assert_called_once_with(cfg)
        self.assertEqual(result, "built")

    def test_runtime_api_dispatches_to_registry(self) -> None:
        fake_runtime = types.SimpleNamespace(
            create=mock.Mock(return_value="rh"),
            infer=mock.Mock(return_value=np.zeros((1, 1), dtype=np.float32)),
            destroy=mock.Mock(return_value=None),
        )
        cfg = RuntimeConfig(
            backend="warboy",
            engine_path="builds/resnet50.enf",
            input_name="input",
            output_name="output",
            input_shape=(1, 3, 224, 224),
        )
        rh = RuntimeHandle(
            backend="warboy",
            engine_path="builds/resnet50.enf",
            input_name="input",
            output_name="output",
            input_shape=(1, 3, 224, 224),
            ctx={},
        )
        x = np.zeros((1, 3, 224, 224), dtype=np.float32)
        with mock.patch("unified_sdk.runtime.api.get_runtime", return_value=fake_runtime):
            created = create_runtime(cfg)
            output = infer(rh, x)
            destroy_runtime(rh)
        fake_runtime.create.assert_called_once_with(cfg)
        fake_runtime.infer.assert_called_once_with(rh, x)
        fake_runtime.destroy.assert_called_once_with(rh)
        self.assertEqual(created, "rh")
        self.assertIsInstance(output, np.ndarray)

    def test_mapping_helpers_report_warboy_backend(self) -> None:
        self.assertEqual(describe_build_api_mapping()["backend"], "warboy")
        self.assertEqual(describe_runtime_api_mapping()["backend"], "warboy")


if __name__ == "__main__":
    unittest.main()

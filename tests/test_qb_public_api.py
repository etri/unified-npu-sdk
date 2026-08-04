from __future__ import annotations

import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_sdk.build.api import build_unified, describe_build_api_mapping
from unified_sdk.frontends import describe_frontend_api_mapping
from unified_sdk.options import QBBuildOptions, QBSequenceRuntimeOptions, QBVisionRuntimeOptions
from unified_sdk.runtime.api import create_runtime, describe_runtime_api_mapping, destroy_runtime, infer
from unified_sdk.sequence_runtime.api import (
    create_sequence_runtime,
    describe_sequence_runtime_api_mapping,
    destroy_sequence_runtime,
    infer_sequence,
)
from unified_sdk.sequence_runtime.types import SequenceBatchParam, SequenceRuntimeConfig
from unified_sdk.types import BuildConfig, RuntimeConfig


class QBPublicAPITests(unittest.TestCase):
    def test_build_api_dispatches_to_registry_builder(self) -> None:
        sentinel = object()
        cfg = BuildConfig(
            model_or_path="models/demo.onnx",
            backend_options=QBBuildOptions(use_random_calib=True),
        )
        with patch("unified_sdk.build.api.get_builder") as get_builder:
            get_builder.return_value.build.return_value = sentinel
            result = build_unified(cfg)
        self.assertIs(result, sentinel)
        get_builder.assert_called_once_with("qb")
        get_builder.return_value.build.assert_called_once_with(cfg)

    def test_runtime_api_dispatches_to_registry_runtime(self) -> None:
        cfg = RuntimeConfig(
            engine_path="demo.mxq",
            input_name="input",
            output_name="output",
            input_shape=(1, 3, 224, 224),
            backend_options=QBVisionRuntimeOptions(),
        )
        fake_handle = SimpleNamespace(backend="qb")
        fake_output = np.zeros((1, 3, 224, 224), dtype=np.float32)
        with patch("unified_sdk.runtime.api.get_runtime") as get_runtime:
            get_runtime.return_value.create.return_value = fake_handle
            get_runtime.return_value.infer.return_value = fake_output
            created = create_runtime(cfg)
            inferred = infer(created, fake_output)
            destroy_runtime(created)
        self.assertIs(created, fake_handle)
        self.assertIs(inferred, fake_output)
        get_runtime.assert_any_call("qb")
        get_runtime.return_value.create.assert_called_once_with(cfg)
        get_runtime.return_value.infer.assert_called_once_with(fake_handle, fake_output)
        get_runtime.return_value.destroy.assert_called_once_with(fake_handle)

    def test_sequence_runtime_api_dispatches_to_registry_runtime(self) -> None:
        cfg = SequenceRuntimeConfig(
            engine_path="demo.mxq",
            input_name="input",
            output_name="output",
            input_shape=(1, 4),
            backend_options=QBSequenceRuntimeOptions(),
        )
        fake_handle = SimpleNamespace(backend="qb")
        fake_input = np.zeros((1, 4), dtype=np.float32)
        fake_params = [SequenceBatchParam(sequence_length=4)]
        with patch("unified_sdk.sequence_runtime.api.get_runtime") as get_runtime:
            get_runtime.return_value.create.return_value = fake_handle
            get_runtime.return_value.infer.return_value = fake_input
            created = create_sequence_runtime(cfg)
            inferred = infer_sequence(created, fake_input, cache_size=1, batch_params=fake_params)
            destroy_sequence_runtime(created)
        self.assertIs(created, fake_handle)
        self.assertIs(inferred, fake_input)
        get_runtime.assert_any_call("qb")
        get_runtime.return_value.create.assert_called_once_with(cfg)
        get_runtime.return_value.infer.assert_called_once_with(
            fake_handle,
            fake_input,
            cache_size=1,
            batch_params=fake_params,
        )
        get_runtime.return_value.destroy.assert_called_once_with(fake_handle)

    def test_mapping_helpers_expose_capability_families(self) -> None:
        self.assertEqual(describe_frontend_api_mapping()["capability_family"], "vision.frontend-prepare-fetch")
        self.assertEqual(describe_build_api_mapping()["capability_family"], "vision.direct-python-compiler")
        self.assertEqual(describe_runtime_api_mapping()["capability_family"], "vision.direct-python-runtime")
        self.assertEqual(
            describe_sequence_runtime_api_mapping()["capability_family"],
            "sequence.low-level-cache-aware-runtime",
        )


if __name__ == "__main__":
    unittest.main()

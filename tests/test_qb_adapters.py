from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_sdk.build.qb_build import _QBBuildAdapter
from unified_sdk.options import QBBuildOptions, QBSequenceRuntimeOptions, QBVisionRuntimeOptions
from unified_sdk.runtime.qb_runtime import _QBVisionRuntime
from unified_sdk.sequence_runtime.qb_sequence_runtime import _QBSequenceRuntime
from unified_sdk.sequence_runtime.types import SequenceBatchParam, SequenceRuntimeConfig
from unified_sdk.types import BuildConfig, RuntimeConfig


class _FakeVisionModel:
    def __init__(self) -> None:
        self.disposed = False

    def infer(self, inputs):
        return np.asarray(inputs[0])

    def dispose(self) -> None:
        self.disposed = True


class _FakeSequenceModel:
    def __init__(self) -> None:
        self.disposed = False
        self.last_params = None
        self.last_cache_size = None

    def infer(self, inputs, params=None, cache_size=None):
        self.last_params = params
        self.last_cache_size = cache_size
        return np.asarray(inputs[0])

    def dispose(self) -> None:
        self.disposed = True


class _FakeModelConfig:
    pass


class _FakeBatchParam:
    def __init__(self, sequence_length: int, cache_size: int, cache_id: int) -> None:
        self.sequence_length = sequence_length
        self.cache_size = cache_size
        self.cache_id = cache_id


class QBAdapterTests(unittest.TestCase):
    def test_build_adapter_places_provided_artifact(self) -> None:
        adapter = _QBBuildAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            src = tmp / "source.mxq"
            src.write_bytes(b"mxq")
            cfg = BuildConfig(
                backend="qb",
                model_or_path=str(src),
                out_dir=tmp / "builds",
                model_name="demo",
                backend_options=QBBuildOptions(),
            )
            result = adapter.build(cfg)
            self.assertTrue(Path(result.compiled_model_path).is_file())
            self.assertEqual(result.meta_data["source"], "provided")

    def test_build_adapter_compiles_with_mocked_vendor_api(self) -> None:
        adapter = _QBBuildAdapter()
        captured: dict[str, object] = {}

        def _fake_compile(**kwargs):
            captured.update(kwargs)
            Path(kwargs["save_path"]).write_bytes(b"mxq")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg = BuildConfig(
                backend="qb",
                model_or_path="models/demo.onnx",
                out_dir=tmp / "builds",
                model_name="demo",
                backend_options=QBBuildOptions(use_random_calib=True),
            )
            with patch(
                "unified_sdk.build.qb_build._resolve_mxq_compile",
                return_value=("qbcompiler", _fake_compile),
            ):
                result = adapter.build(cfg)

            self.assertTrue(Path(result.compiled_model_path).is_file())
            self.assertEqual(captured["target_device"], "aries-rb")
            self.assertEqual(result.meta_data["source"], "qbcompiler_compile")

    def test_vision_runtime_adapter_with_mocked_vendor_runtime(self) -> None:
        adapter = _QBVisionRuntime()
        fake_model = _FakeVisionModel()
        fake_qb_model = SimpleNamespace(load=lambda path, model_config: fake_model)
        fake_qb_type = SimpleNamespace(ModelConfig=_FakeModelConfig)

        cfg = RuntimeConfig(
            backend="qb",
            engine_path="demo.mxq",
            input_name="input",
            output_name="output",
            input_shape=(1, 3, 224, 224),
            backend_options=QBVisionRuntimeOptions(),
        )

        with patch("unified_sdk.runtime.qb_runtime.validate_mxq_path", return_value=Path("demo.mxq")), patch(
            "unified_sdk.runtime.qb_runtime.load_qbruntime_modules",
            return_value=(SimpleNamespace(), fake_qb_model, fake_qb_type),
        ):
            rh = adapter.create(cfg)
            output = adapter.infer(rh, np.zeros((1, 3, 224, 224), dtype=np.float32))
            adapter.destroy(rh)

        self.assertEqual(output.shape, (1, 3, 224, 224))
        self.assertTrue(fake_model.disposed)
        self.assertEqual(rh.ctx, {})

    def test_sequence_runtime_adapter_normalizes_batch_params(self) -> None:
        adapter = _QBSequenceRuntime()
        fake_model = _FakeSequenceModel()
        fake_qb_model = SimpleNamespace(load=lambda path, model_config: fake_model)
        fake_qbruntime = SimpleNamespace(BatchParam=_FakeBatchParam)
        fake_qb_type = SimpleNamespace(ModelConfig=_FakeModelConfig)

        cfg = SequenceRuntimeConfig(
            backend="qb",
            engine_path="demo.mxq",
            input_name="input",
            output_name="output",
            input_shape=(1,),
            backend_options=QBSequenceRuntimeOptions(allow_dynamic_shape=True),
        )

        with patch("unified_sdk.sequence_runtime.qb_sequence_runtime.validate_mxq_path", return_value=Path("demo.mxq")), patch(
            "unified_sdk.sequence_runtime.qb_sequence_runtime.load_qbruntime_modules",
            return_value=(fake_qbruntime, fake_qb_model, fake_qb_type),
        ):
            rh = adapter.create(cfg)
            output = adapter.infer(
                rh,
                np.zeros((1, 4), dtype=np.float32),
                batch_params=[SequenceBatchParam(sequence_length=4, cache_size=2, cache_id=1)],
            )
            adapter.destroy(rh)

        self.assertEqual(output.shape, (1, 4))
        self.assertIsNotNone(fake_model.last_params)
        assert fake_model.last_params is not None
        self.assertEqual(fake_model.last_params[0].sequence_length, 4)
        self.assertEqual(fake_model.last_params[0].cache_size, 2)
        self.assertEqual(fake_model.last_params[0].cache_id, 1)
        self.assertTrue(fake_model.disposed)


if __name__ == "__main__":
    unittest.main()

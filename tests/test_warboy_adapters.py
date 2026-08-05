from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_sdk.build.warboy_build import _WarboyBuildAdapter  # noqa: E402
from unified_sdk.frontends.types import PreparedWarboyBuildInput, PreparedWarboyCompileSource, ProvidedWarboyArtifact  # noqa: E402
from unified_sdk.options import WarboyBuildOptions, WarboyRuntimeOptions  # noqa: E402
from unified_sdk.runtime.warboy_runtime import _WarboyRuntime  # noqa: E402
from unified_sdk.types import BuildConfig, RuntimeConfig  # noqa: E402


class _FakeRunner:
    def __init__(self) -> None:
        self.closed = False

    def run(self, inputs):
        if isinstance(inputs, list):
            return [np.asarray(inputs[0])]
        return np.asarray(inputs)

    def close(self) -> None:
        self.closed = True


class WarboyAdaptersTest(unittest.TestCase):
    def test_build_adapter_places_provided_enf(self) -> None:
        adapter = _WarboyBuildAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src = tmp_path / "source.enf"
            src.write_text("enf")
            Path(f"{src}.json").write_text(json.dumps({"input_contract": {"input_dtype": "uint8"}}))
            cfg = BuildConfig(
                backend="warboy",
                model_or_path=str(src),
                out_dir=str(tmp_path / "builds"),
                model_name="resnet50",
                input_name="input",
                input_shape=(1, 3, 224, 224),
                backend_options=WarboyBuildOptions(),
            )
            result = adapter.build(cfg)
            self.assertTrue(Path(result.compiled_model_path).is_file())
            self.assertEqual(result.meta_data["source"], "provided")
            self.assertEqual(result.meta_data["input_contract"]["input_dtype"], "uint8")
            self.assertTrue(Path(f"{result.compiled_model_path}.json").is_file())

    def test_build_adapter_prefers_prepared_input_contract(self) -> None:
        adapter = _WarboyBuildAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            src = tmp_path / "prepared.enf"
            src.write_text("enf")
            cfg = BuildConfig(
                backend="warboy",
                model_or_path="should_not_be_reparsed.bin",
                out_dir=str(tmp_path / "builds"),
                model_name="resnet50",
                input_name="input",
                input_shape=(1, 3, 224, 224),
                backend_options=WarboyBuildOptions(),
                prepared_input=PreparedWarboyBuildInput(
                    kind="provided_artifact",
                    provided_artifact=ProvidedWarboyArtifact(
                        source_path=src,
                        destination_path=tmp_path / "builds" / "resnet50.enf",
                    ),
                ),
            )
            result = adapter.build(cfg)
            self.assertTrue(Path(result.compiled_model_path).is_file())
            self.assertEqual(result.meta_data["origin"], str(src))

    def test_build_adapter_runs_compiler_for_quantized_onnx(self) -> None:
        adapter = _WarboyBuildAdapter()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            onnx_path = tmp_path / "resnet50_quantized.onnx"
            onnx_path.write_text("onnx")

            def _fake_run(command, check, text, capture_output):
                out_path = Path(command[command.index("-o") + 1])
                out_path.write_text("compiled-enf")
                return types.SimpleNamespace(returncode=0, stdout="", stderr="")

            cfg = BuildConfig(
                backend="warboy",
                model_or_path=str(onnx_path),
                out_dir=str(tmp_path / "builds"),
                model_name="resnet50",
                input_name="input",
                input_shape=(1, 3, 224, 224),
                backend_options=WarboyBuildOptions(target_npu="warboy"),
            )
            with mock.patch("unified_sdk.build.warboy_build.shutil.which", return_value="/usr/bin/furiosa-compiler"):
                with mock.patch("unified_sdk.build.warboy_build.subprocess.run", side_effect=_fake_run):
                    with mock.patch(
                        "unified_sdk.build.warboy_build._inspect_onnx_input_contract",
                        return_value={"input_dtype": "float32", "input_shape": [1, 3, 224, 224], "inspection_warning": None},
                    ):
                        result = adapter.build(cfg)
            self.assertTrue(Path(result.compiled_model_path).is_file())
            self.assertEqual(result.meta_data["target_npu"], "warboy")
            self.assertEqual(result.meta_data["input_contract"]["input_dtype"], "float32")
            sidecar = Path(f"{result.compiled_model_path}.json")
            self.assertTrue(sidecar.is_file())

    def test_runtime_adapter_uses_backend_options(self) -> None:
        adapter = _WarboyRuntime()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            enf_path = tmp_path / "resnet50.enf"
            enf_path.write_text("enf")
            fake_sync = types.SimpleNamespace(create_runner=mock.Mock(return_value=_FakeRunner()))
            fake_runtime = types.ModuleType("furiosa.runtime")
            fake_runtime.sync = fake_sync
            fake_furiosa = types.ModuleType("furiosa")
            fake_furiosa.runtime = fake_runtime

            cfg = RuntimeConfig(
                backend="warboy",
                engine_path=str(enf_path),
                input_name="input",
                output_name="output",
                input_shape=(1, 3, 224, 224),
                backend_options=WarboyRuntimeOptions(device="warboy(0)*2", allow_dynamic_shape=False),
            )
            with mock.patch.dict(sys.modules, {"furiosa": fake_furiosa, "furiosa.runtime": fake_runtime}):
                rh = adapter.create(cfg)
                out = adapter.infer(rh, np.zeros((1, 3, 224, 224), dtype=np.float32))
                adapter.destroy(rh)
            fake_sync.create_runner.assert_called_once_with(str(enf_path), device="warboy(0)*2")
            self.assertIsInstance(out, np.ndarray)

    def test_runtime_adapter_returns_list_for_multi_output(self) -> None:
        adapter = _WarboyRuntime()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            enf_path = tmp_path / "multi.enf"
            enf_path.write_text("enf")

            class _MultiRunner(_FakeRunner):
                def run(self, inputs):
                    arr = np.asarray(inputs[0])
                    return [arr, arr]

            fake_sync = types.SimpleNamespace(create_runner=mock.Mock(return_value=_MultiRunner()))
            fake_runtime = types.ModuleType("furiosa.runtime")
            fake_runtime.sync = fake_sync
            fake_furiosa = types.ModuleType("furiosa")
            fake_furiosa.runtime = fake_runtime

            cfg = RuntimeConfig(
                backend="warboy",
                engine_path=str(enf_path),
                input_name="input",
                input_shape=(1, 3, 224, 224),
                backend_options=WarboyRuntimeOptions(),
            )
            with mock.patch.dict(sys.modules, {"furiosa": fake_furiosa, "furiosa.runtime": fake_runtime}):
                rh = adapter.create(cfg)
                out = adapter.infer(rh, np.zeros((1, 3, 224, 224), dtype=np.float32))
                adapter.destroy(rh)
            self.assertIsInstance(out, list)
            self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()

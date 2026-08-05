from __future__ import annotations

import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_sdk.frontends import (  # noqa: E402
    WarboyFrontendBuildRequest,
    prepare_warboy_build_input,
    resolve_warboy_build_request,
)


class WarboyFrontendsTest(unittest.TestCase):
    def test_prepare_warboy_build_input_for_provided_enf(self) -> None:
        prepared = prepare_warboy_build_input("models/resnet50.enf", "builds/resnet50.enf")
        self.assertEqual(prepared.kind, "provided_artifact")
        self.assertIsNotNone(prepared.provided_artifact)
        self.assertTrue(str(prepared.provided_artifact.destination_path).endswith("resnet50.enf"))

    def test_resolve_warboy_build_request_for_quantized_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            onnx_path = models_dir / "resnet50_quantized.onnx"
            onnx_path.write_text("dummy")
            request = WarboyFrontendBuildRequest(
                model_name="resnet50",
                models_dir=models_dir,
                target_npu="warboy-2pe",
                from_onnx=onnx_path,
            )
            resolved = resolve_warboy_build_request(request=request)
            self.assertEqual(resolved.kind, "quantized_onnx")
            self.assertEqual(resolved.prepared_input.kind, "compile_source")
            self.assertEqual(resolved.model_or_path, str(onnx_path.resolve()))

    def test_resolve_warboy_build_request_prefers_local_enf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            local_enf = models_dir / "resnet50.enf"
            local_enf.write_text("enf")
            request = WarboyFrontendBuildRequest(
                model_name="resnet50",
                models_dir=models_dir,
                target_npu="warboy-2pe",
            )
            frontend_module = importlib.import_module("unified_sdk.frontends.resolve_warboy_build_request")
            with mock.patch.object(
                frontend_module,
                "fetch_model_zoo_enf",
                side_effect=AssertionError("model zoo fetch should not run when local ENF exists"),
            ):
                resolved = resolve_warboy_build_request(request=request)
            self.assertEqual(resolved.kind, "local_enf")
            self.assertEqual(resolved.prepared_input.kind, "provided_artifact")


if __name__ == "__main__":
    unittest.main()

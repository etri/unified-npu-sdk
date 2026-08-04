from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_sdk.frontends import (
    QBFrontendBuildRequest,
    describe_frontend_api_mapping,
    prepare_qb_build_input,
    resolve_qb_build_request,
)


class QBFrontendTests(unittest.TestCase):
    def test_prepare_qb_build_input_classifies_provided_mxq(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "demo.mxq"
            src.write_bytes(b"mxq")
            prepared = prepare_qb_build_input(src, Path(tmpdir) / "out.mxq")
            self.assertEqual(prepared.kind, "provided_artifact")
            self.assertIsNotNone(prepared.provided_artifact)
            assert prepared.provided_artifact is not None
            self.assertEqual(prepared.provided_artifact.source_path, src.resolve())

    def test_prepare_qb_build_input_classifies_compile_source(self) -> None:
        prepared = prepare_qb_build_input("models/demo.onnx", "builds/demo.mxq")
        self.assertEqual(prepared.kind, "compile_source")
        self.assertIsNotNone(prepared.compile_source)

    def test_resolve_qb_build_request_from_onnx(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            onnx_path = root / "demo.onnx"
            onnx_path.write_bytes(b"onnx")
            resolved = resolve_qb_build_request(
                request=QBFrontendBuildRequest(
                    model_name="demo",
                    models_dir=root / "models",
                    product="aries",
                    core_mode="global8",
                    from_onnx=onnx_path,
                )
            )
            self.assertEqual(resolved.model_or_path, str(onnx_path.resolve()))
            self.assertIn("compiler Python API compile", resolved.source_description)
            self.assertEqual(resolved.kind, "local_onnx")

    def test_describe_frontend_api_mapping_reports_prepare_capability(self) -> None:
        mapping = describe_frontend_api_mapping()
        self.assertEqual(mapping["capability_family"], "vision.frontend-prepare-fetch")
        self.assertIn("resolve_qb_build_request", mapping["unified_frontend_api"])


if __name__ == "__main__":
    unittest.main()

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
    RNGDFrontendBuildRequest,
    describe_frontend_api_mapping,
    resolve_rngd_build_request,
)


class RNGDFrontendTests(unittest.TestCase):
    def test_resolve_fetch_request_for_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = resolve_rngd_build_request(
                request=RNGDFrontendBuildRequest(
                    model_or_path="furiosa-ai/Qwen2.5-0.5B-Instruct",
                    out_dir=Path(tmpdir),
                    model_name="demo",
                    build_mode="fetch",
                )
            )
        self.assertEqual(resolved.kind, "model_id")
        self.assertIsNone(resolved.output_path)

    def test_resolve_fxb_build_request_for_local_model_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = root / "model"
            model_dir.mkdir()
            resolved = resolve_rngd_build_request(
                request=RNGDFrontendBuildRequest(
                    model_or_path=model_dir,
                    out_dir=root / "artifacts",
                    model_name="demo",
                    build_mode="fxb_build",
                )
            )
        self.assertEqual(resolved.kind, "fxb_build_source")
        self.assertIsNotNone(resolved.output_path)
        assert resolved.output_path is not None
        self.assertEqual(resolved.output_path.suffix, ".fxb")

    def test_resolve_fxb_build_rejects_prebuilt_artifact_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_dir = root / "artifact"
            artifact_dir.mkdir()
            (artifact_dir / "artifact.json").write_text("{}")
            with self.assertRaises(RuntimeError):
                resolve_rngd_build_request(
                    request=RNGDFrontendBuildRequest(
                        model_or_path=artifact_dir,
                        out_dir=root / "artifacts",
                        model_name="demo",
                        build_mode="fxb_build",
                    )
                )

    def test_describe_frontend_api_mapping_reports_prepare_capability(self) -> None:
        mapping = describe_frontend_api_mapping()
        self.assertEqual(mapping["capability_family"], "llm.frontend-prepare-fetch")
        self.assertIn("resolve_rngd_build_request", mapping["unified_frontend_api"])


if __name__ == "__main__":
    unittest.main()

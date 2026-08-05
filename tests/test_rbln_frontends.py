from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_sdk.frontends import (  # noqa: E402
    RBLNLLMFrontendBuildRequest,
    RBLNVisionFrontendBuildRequest,
    list_model_zoo_targets,
    resolve_rbln_llm_build_request,
    prepare_rbln_vision_build_input,
    resolve_rbln_vision_build_request,
)


class RBLNFrontendsTest(unittest.TestCase):
    def test_list_model_zoo_targets_contains_resnet50(self) -> None:
        targets = list_model_zoo_targets()
        self.assertIn("resnet50", targets)

    def test_prepare_build_input_for_provided_rbln(self) -> None:
        prepared = prepare_rbln_vision_build_input("models/resnet50.rbln", "builds/resnet50.rbln")
        self.assertEqual(prepared.kind, "provided_artifact")
        self.assertTrue(str(prepared.provided_artifact.destination_path).endswith("resnet50.rbln"))

    def test_resolve_request_for_pretrained_model_uses_optimum_frontend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            resolved = resolve_rbln_vision_build_request(
                RBLNVisionFrontendBuildRequest(
                    model_name="resnet50",
                    models_dir=models_dir,
                    model_zoo_model="resnet50",
                    pretrained=True,
                )
            )
        self.assertEqual(resolved.kind, "optimum_source_model")
        self.assertEqual(resolved.prepared_input.compile_source.compile_frontend, "optimum_image_classification")

    def test_resolve_request_for_missing_local_compiled_ref_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            with self.assertRaises(FileNotFoundError):
                resolve_rbln_vision_build_request(
                    RBLNVisionFrontendBuildRequest(
                        model_name="resnet50",
                        models_dir=models_dir,
                        compiled_model_ref="builds/missing_artifact",
                    )
                )

    def test_resolve_request_for_local_compiled_dir_stays_provided_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            models_dir = root / "models"
            compiled_dir = root / "builds" / "compiled_resnet50"
            compiled_dir.mkdir(parents=True)
            (compiled_dir / "resnet50.rbln").write_text("rbln")
            (compiled_dir / "rbln_config.json").write_text("{}")

            resolved = resolve_rbln_vision_build_request(
                RBLNVisionFrontendBuildRequest(
                    model_name="resnet50",
                    models_dir=models_dir,
                    compiled_model_ref=str(compiled_dir),
                )
            )

        self.assertEqual(resolved.kind, "compiled_dir")
        self.assertEqual(resolved.prepared_input.kind, "provided_artifact")
        self.assertEqual(resolved.prepared_input.provided_artifact.source_path.name, "compiled_resnet50")

    def test_resolve_request_for_weights_path(self) -> None:
        class _FakeModel:
            def eval(self):
                return self

            def load_state_dict(self, state_dict, strict=False):
                self.state_dict = state_dict
                self.strict = strict

        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir)
            weights = models_dir / "resnet50.pth"
            weights.write_text("placeholder")

            with mock.patch.dict(sys.modules, {"torch": mock.Mock()}):
                with mock.patch("unified_sdk.frontends._build_torchvision_resnet50", return_value=_FakeModel()):
                    with mock.patch("unified_sdk.frontends._load_state_dict", return_value={"layer.weight": 1}):
                        resolved = resolve_rbln_vision_build_request(
                            RBLNVisionFrontendBuildRequest(
                                model_name="resnet50",
                                models_dir=models_dir,
                                weights_path=weights,
                            )
                        )

        self.assertEqual(resolved.kind, "torch_model")
        self.assertEqual(resolved.prepared_input.compile_source.source_label, "torch_model")

    def test_resolve_llm_fetch_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = resolve_rbln_llm_build_request(
                RBLNLLMFrontendBuildRequest(
                    model_ref="Qwen/Qwen3-0.6B",
                    out_dir=Path(tmpdir),
                    model_name="qwen3",
                    build_mode="fetch",
                )
            )
        self.assertEqual(resolved.kind, "runtime_model_ref")
        self.assertEqual(resolved.prepared_input.kind, "runtime_model_ref")

    def test_resolve_llm_artifact_build_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = resolve_rbln_llm_build_request(
                RBLNLLMFrontendBuildRequest(
                    model_ref="Qwen/Qwen3-0.6B",
                    out_dir=Path(tmpdir),
                    model_name="qwen3",
                    build_mode="optimum_compile",
                )
            )
        self.assertEqual(resolved.kind, "artifact_build")
        self.assertEqual(resolved.prepared_input.kind, "artifact_build")
        self.assertTrue(str(resolved.prepared_input.artifact_dir).endswith("qwen3"))


if __name__ == "__main__":
    unittest.main()

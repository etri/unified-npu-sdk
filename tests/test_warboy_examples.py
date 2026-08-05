from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import runpy
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_sdk.frontends import (  # noqa: E402
    PreparedWarboyBuildInput,
    ProvidedWarboyArtifact,
    ResolvedWarboyBuildRequest,
)


class WarboyExamplesTest(unittest.TestCase):
    def test_runtime_loop_always_destroys_runtime_on_error(self) -> None:
        script_path = REPO_ROOT / "examples" / "run_warboy_infer.py"
        module_globals = runpy.run_path(str(script_path), run_name="warboy_infer_test")
        run_runtime_loop = module_globals["_run_runtime_loop"]

        fake_handle = object()
        created = []
        destroyed = []

        def _create_runtime(_cfg):
            created.append(True)
            return fake_handle

        def _infer(_rh, _batch):
            raise RuntimeError("boom")

        def _destroy_runtime(rh):
            destroyed.append(rh)

        with self.assertRaises(RuntimeError):
            run_runtime_loop(
                cfg=object(),
                batch=object(),
                iters=1,
                create_runtime_fn=_create_runtime,
                infer_fn=_infer,
                destroy_runtime_fn=_destroy_runtime,
            )

        self.assertTrue(created)
        self.assertEqual(destroyed, [fake_handle])

    def test_run_warboy_build_uses_keyword_frontend_request(self) -> None:
        script_path = REPO_ROOT / "examples" / "run_warboy_build.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            provided_enf = tmp_path / "resnet50.enf"
            provided_enf.write_text("enf")
            out_dir = tmp_path / "builds"

            def _fake_resolve(*, request):
                return ResolvedWarboyBuildRequest(
                    model_or_path=str(provided_enf),
                    source_description=f"provided .enf: {provided_enf}",
                    kind="provided_artifact",
                    prepared_input=PreparedWarboyBuildInput(
                        kind="provided_artifact",
                        provided_artifact=ProvidedWarboyArtifact(
                            source_path=provided_enf,
                            destination_path=out_dir / "resnet50.enf",
                        ),
                    ),
                )

            fake_result = types.SimpleNamespace(compiled_model_path=str(out_dir / "resnet50.enf"))
            captured_cfg = {}
            argv = [
                str(script_path),
                "--models-dir",
                str(tmp_path),
                "--out-dir",
                str(out_dir),
                "--enf",
                str(provided_enf),
                "--model-name",
                "resnet50",
            ]

            def _fake_build(cfg):
                captured_cfg["cfg"] = cfg
                return fake_result

            stdout = StringIO()
            with mock.patch("unified_sdk.frontends.resolve_warboy_build_request", side_effect=_fake_resolve):
                with mock.patch("unified_sdk.build.build_unified", side_effect=_fake_build):
                    with mock.patch.object(sys, "argv", argv):
                        with redirect_stdout(stdout):
                            runpy.run_path(str(script_path), run_name="__main__")

            self.assertIn("Complete!", stdout.getvalue())
            self.assertIsNotNone(captured_cfg["cfg"].prepared_input)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import runpy
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

            stdout = StringIO()
            with mock.patch("unified_sdk.frontends.resolve_warboy_build_request", side_effect=_fake_resolve):
                with mock.patch("unified_sdk.build.build_unified", return_value=fake_result):
                    with mock.patch.object(sys, "argv", argv):
                        with redirect_stdout(stdout):
                            runpy.run_path(str(script_path), run_name="__main__")

            self.assertIn("Complete!", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from unified_sdk.build.api import build_unified_LLM, describe_build_api_mapping
from unified_sdk.frontends import describe_frontend_api_mapping
from unified_sdk.options import RNGDBuildOptions, RNGDRuntimeOptions
from unified_sdk.runtime.api import (
    create_runtime_LLM,
    describe_runtime_api_mapping,
    destroy_runtime_LLM,
    generate_LLM,
    infer_LLM,
)
from unified_sdk.types import LLMBuildConfig, LLMRuntimeConfig


class RNGDPublicAPITests(unittest.TestCase):
    def test_build_api_dispatches_to_registry_builder(self) -> None:
        sentinel = object()
        cfg = LLMBuildConfig(
            model_or_path="furiosa-ai/Qwen2.5-0.5B-Instruct",
            backend_options=RNGDBuildOptions(),
        )
        with patch("unified_sdk.build.api.get_builder") as get_builder:
            get_builder.return_value.build.return_value = sentinel
            result = build_unified_LLM(cfg)
        self.assertIs(result, sentinel)
        get_builder.assert_called_once_with("rngd")
        get_builder.return_value.build.assert_called_once_with(cfg)

    def test_runtime_api_dispatches_to_registry_runtime(self) -> None:
        cfg = LLMRuntimeConfig(
            engine_path="furiosa-ai/Qwen2.5-0.5B-Instruct",
            backend_options=RNGDRuntimeOptions(),
        )
        fake_handle = SimpleNamespace(backend="rngd")
        with patch("unified_sdk.runtime.api.get_runtime") as get_runtime:
            get_runtime.return_value.create.return_value = fake_handle
            get_runtime.return_value.infer.return_value = "infer"
            get_runtime.return_value.generate.return_value = "generate"
            created = create_runtime_LLM(cfg)
            inferred = infer_LLM(created, "hello")
            generated = generate_LLM(created, "hello")
            destroy_runtime_LLM(created)
        self.assertIs(created, fake_handle)
        self.assertEqual(inferred, "infer")
        self.assertEqual(generated, "generate")
        get_runtime.assert_any_call("rngd")

    def test_mapping_helpers_expose_capability_families(self) -> None:
        self.assertEqual(describe_frontend_api_mapping()["capability_family"], "llm.frontend-prepare-fetch")
        self.assertEqual(describe_build_api_mapping()["capability_family"], "llm.fxb-and-generation")
        self.assertEqual(describe_runtime_api_mapping()["capability_family"], "llm.artifact-and-generation-runtime")


if __name__ == "__main__":
    unittest.main()

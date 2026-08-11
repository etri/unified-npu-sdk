from __future__ import annotations

import argparse
import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path


def _is_repo_root(path: Path) -> bool:
    return (path / "src" / "unified_sdk").is_dir() and (path / "examples").is_dir()


def _resolve_repo_root() -> Path:
    env_root = os.getenv("UNIFIED_SDK_REPO_ROOT")
    if env_root:
        candidate = Path(env_root).resolve()
        if _is_repo_root(candidate):
            return candidate

    cwd = Path.cwd().resolve()
    if _is_repo_root(cwd):
        return cwd

    file_root = Path(__file__).resolve().parents[2]
    if _is_repo_root(file_root):
        return file_root

    for candidate in (Path("/workspace/unified-sdk"), Path("/workspace/unified-npu-sdk")):
        if _is_repo_root(candidate):
            return candidate

    return file_root


REPO_ROOT = _resolve_repo_root()


def _vendor_root_candidates(repo_root: Path) -> list[Path]:
    candidates: list[Path] = []
    env_root = os.getenv("TENSORRT_LLM_SRC")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    candidates.extend(
        [
            repo_root.parent / "TensorRT-LLM",
            repo_root / "TensorRT-LLM",
            Path("/workspace/TensorRT-LLM"),
            Path("/workspace/tensorrt_llm"),
            Path("/opt/TensorRT-LLM"),
        ]
    )

    seen: set[str] = set()
    unique: list[Path] = []
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _find_vendor_convert_script(repo_root: Path) -> Path:
    for root in _vendor_root_candidates(repo_root):
        script = root / "examples" / "llama" / "convert_checkpoint.py"
        if script.is_file():
            return script.resolve()
    raise FileNotFoundError(
        "Could not find TensorRT-LLM's examples/llama/convert_checkpoint.py. "
        "Set TENSORRT_LLM_SRC to a matching TensorRT-LLM source checkout, or place the checkout in a common location "
        f"such as {repo_root.parent / 'TensorRT-LLM'}."
    )


def _import_llama_class():
    candidates = (
        ("tensorrt_llm.models", "LLaMAForCausalLM"),
        ("tensorrt_llm.models", "LlamaForCausalLM"),
        ("tensorrt_llm.models.llama", "LLaMAForCausalLM"),
        ("tensorrt_llm.models.llama", "LlamaForCausalLM"),
        ("tensorrt_llm.models.llama.model", "LLaMAForCausalLM"),
        ("tensorrt_llm.models.llama.model", "LlamaForCausalLM"),
        ("tensorrt_llm._torch.models", "LLaMAForCausalLM"),
        ("tensorrt_llm._torch.models", "LlamaForCausalLM"),
        ("tensorrt_llm._torch.models.modeling_llama", "LLaMAForCausalLM"),
        ("tensorrt_llm._torch.models.modeling_llama", "LlamaForCausalLM"),
    )
    attempted: list[str] = []
    for module_name, attr_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            attempted.append(f"{module_name}.{attr_name} (import failed: {exc})")
            continue
        cls = getattr(module, attr_name, None)
        if cls is not None:
            return cls
        attempted.append(f"{module_name}.{attr_name} (missing)")

    discovery_modules = (
        "tensorrt_llm.models",
        "tensorrt_llm.models.llama.model",
        "tensorrt_llm._torch.models",
        "tensorrt_llm._torch.models.modeling_llama",
    )
    for module_name in discovery_modules:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            attempted.append(f"{module_name} (discovery import failed: {exc})")
            continue
        for attr_name in dir(module):
            if "llama" not in attr_name.lower():
                continue
            obj = getattr(module, attr_name, None)
            if inspect.isclass(obj) and callable(getattr(obj, "from_hugging_face", None)):
                return obj
        attempted.append(f"{module_name} (no llama-like class with from_hugging_face)")

    raise ImportError(
        "Could not import a LLaMA/TinyLlama conversion class from installed tensorrt_llm package. "
        f"Attempted probes: {attempted}"
    )


def _import_mapping_class():
    candidates = (
        ("tensorrt_llm.mapping", "Mapping"),
        ("tensorrt_llm.models", "Mapping"),
    )
    for module_name, attr_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        cls = getattr(module, attr_name, None)
        if cls is not None:
            return cls
    raise ImportError("Could not import Mapping from installed tensorrt_llm package")


def _build_mapping(args):
    tp_size = args.tp_size or 1
    pp_size = args.pp_size or 1
    if tp_size == 1 and pp_size == 1:
        return None
    Mapping = _import_mapping_class()
    world_size = tp_size * pp_size
    return Mapping(world_size=world_size, rank=0, tp_size=tp_size, pp_size=pp_size)


def _call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    supported = {name: value for name, value in kwargs.items() if name in sig.parameters and value is not None}
    return fn(**supported)


def _convert_with_installed_api(args) -> bool:
    try:
        import tensorrt_llm  # noqa: F401
    except Exception:
        return False

    LLaMAForCausalLM = _import_llama_class()
    mapping = _build_mapping(args)
    model_dir = str(Path(args.model_dir).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    from_hf = getattr(LLaMAForCausalLM, "from_hugging_face", None)
    if not callable(from_hf):
        raise RuntimeError("Installed TensorRT-LLM LLaMA class does not expose from_hugging_face()")

    from_hf_kwargs = {
        "hf_model_dir": model_dir,
        "model_dir": model_dir,
        "dtype": args.dtype,
        "mapping": mapping,
        "load_by_shard": args.load_by_shard,
    }
    model = _call_with_supported_kwargs(from_hf, **from_hf_kwargs)
    save_checkpoint = getattr(model, "save_checkpoint", None)
    if not callable(save_checkpoint):
        raise RuntimeError("Installed TensorRT-LLM LLaMA model object does not expose save_checkpoint()")

    save_kwargs = {
        "output_dir": str(output_dir),
        "save_dir": str(output_dir),
        "save_config": True,
    }
    _call_with_supported_kwargs(save_checkpoint, **save_kwargs)
    return True


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified SDK wrapper for TensorRT-LLM's LLaMA/TinyLlama convert_checkpoint.py. "
            "It keeps the smoke flow under unified-sdk/examples while delegating to a matching vendor source checkout."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tp_size", type=int, default=None)
    parser.add_argument("--pp_size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--load_by_shard", action="store_true")
    return parser


def main() -> int:
    parser = _build_parser()
    args, passthrough = parser.parse_known_args()
    vendor_script = None
    try:
        used_installed_api = _convert_with_installed_api(args)
    except Exception as exc:
        raise RuntimeError(
            "TensorRT-LLM checkpoint prepare via installed Python API failed. "
            "If you intended to use a vendor source checkout fallback, ensure it matches the installed release. "
            f"Original error: {exc}"
        ) from exc
    if not used_installed_api:
        vendor_script = _find_vendor_convert_script(REPO_ROOT)

    print("== TensorRT-LLM checkpoint prepare ==")
    print(f"repo_root = {REPO_ROOT}")
    print(f"model_dir = {args.model_dir}")
    print(f"output_dir = {args.output_dir}")
    if used_installed_api:
        print("conversion_api = installed tensorrt_llm Python API")
        return 0

    cmd = [
        sys.executable,
        str(vendor_script),
        "--model_dir",
        args.model_dir,
        "--output_dir",
        args.output_dir,
        "--dtype",
        args.dtype,
    ]
    if args.tp_size is not None:
        cmd.extend(["--tp_size", str(args.tp_size)])
    if args.pp_size is not None:
        cmd.extend(["--pp_size", str(args.pp_size)])
    if args.workers is not None:
        cmd.extend(["--workers", str(args.workers)])
    if args.load_by_shard:
        cmd.append("--load_by_shard")
    cmd.extend(passthrough)
    print(f"conversion_api = vendor source fallback: {vendor_script}")
    completed = subprocess.run(cmd)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

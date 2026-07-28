# examples/inspect_qb_model.py
"""
QB 컴파일 결과(.mxq) 파일의 요약 정보를 출력합니다.
qbruntime.type.get_model_summary 를 우선 사용하고, 없으면 `mobilint-cli mxqtool show`
로 폴백합니다.
"""
import argparse
import subprocess
from pathlib import Path
import sys
import os

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

    file_root = Path(__file__).resolve().parents[1]
    if _is_repo_root(file_root):
        return file_root

    for candidate in (Path("/workspace/unified-sdk"), Path("/workspace/unified-npu-sdk")):
        if _is_repo_root(candidate):
            return candidate

    return file_root


REPO_ROOT = _resolve_repo_root()
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a compiled Mobilint ARISE (QB) .mxq model.")
    parser.add_argument("model_path", nargs="?", default=str(REPO_ROOT / "builds" / "resnet50.mxq"))
    parser.add_argument("--device", type=int, default=int(os.getenv("MBLT_DEVICE", "0")))
    return parser


def _inspect_via_qbruntime(p: Path) -> bool:
    try:
        from qbruntime import type as qb_type
    except Exception as e:
        print(f"[qbruntime unavailable] {type(e).__name__}: {e}")
        return False

    ok = False
    summary_fn = getattr(qb_type, "get_model_summary", None)
    if callable(summary_fn):
        try:
            print("\n== QB model summary (qbruntime.type.get_model_summary) ==")
            print(summary_fn(str(p)))
            ok = True
        except Exception as e:
            print(f"[get_model_summary failed] {type(e).__name__}: {e}")

    dev_fn = getattr(qb_type, "get_available_device_numbers", None)
    if callable(dev_fn):
        try:
            print("available_devices =", dev_fn())
        except Exception as e:
            print(f"[get_available_device_numbers failed] {type(e).__name__}: {e}")
    return ok


def _inspect_via_cli(p: Path) -> bool:
    try:
        completed = subprocess.run(
            ["mobilint-cli", "mxqtool", "show", str(p)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        print("Error: neither qbruntime nor 'mobilint-cli' is available to inspect the model.")
        return False
    print("\n== QB model summary (mobilint-cli mxqtool show) ==")
    print(completed.stdout.rstrip())
    return completed.returncode == 0


def inspect(model_path: str, *, device: int) -> None:
    p = Path(model_path)
    if not p.is_file():
        print(f"Error: file not found - {p}")
        return
    if p.suffix != ".mxq":
        print(f"Error: expected a .mxq file - {p}")
        return

    print(f"\n== QB model: {p.name} ==")
    print(f"  path: {p}")
    print(f"  device: {device}")

    if _inspect_via_qbruntime(p):
        return
    _inspect_via_cli(p)


if __name__ == "__main__":
    args = _build_parser().parse_args()
    inspect(args.model_path, device=args.device)

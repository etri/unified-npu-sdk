# examples/inspect_warboy_model.py
"""
Warboy 컴파일 결과(.enf) 파일의 입출력 텐서 정보를 출력합니다.
furiosa.runtime 의 runner 에서 사용 가능한 메타 정보를 best-effort로 덤프합니다.
"""
import argparse
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
    parser = argparse.ArgumentParser(description="Inspect a compiled FuriosaAI Warboy .enf model.")
    parser.add_argument("model_path", nargs="?", default=str(REPO_ROOT / "builds" / "resnet50.enf"))
    parser.add_argument("--device", default=os.getenv("FURIOSA_DEVICES", None))
    return parser


def inspect(model_path: str, *, device) -> None:
    try:
        from furiosa.runtime import sync
    except ImportError:
        print("Error: 'furiosa.runtime' not found. Install furiosa-sdk first (developer.furiosa.ai).")
        return

    p = Path(model_path)
    if not p.is_file():
        print(f"Error: file not found - {p}")
        return
    if p.suffix != ".enf":
        print(f"Error: expected a .enf file - {p}")
        return

    try:
        runner = sync.create_runner(str(p), device=str(device)) if device else sync.create_runner(str(p))
    except TypeError:
        runner = sync.create_runner(str(p))
    except Exception as e:
        print(f"Error: failed to create runner ({type(e).__name__}): {e}")
        return

    print(f"\n== Warboy model: {p.name} ==")
    print(f"  path: {p}")
    print(f"  device: {device}")

    # furiosa runner API 는 버전에 따라 노출 속성이 다르므로 best-effort 출력
    try:
        for attr in ("model", "inputs", "outputs", "input_num", "output_num"):
            if hasattr(runner, attr):
                value = getattr(runner, attr)
                try:
                    value = value() if callable(value) else value
                except Exception as e:
                    value = f"<{type(e).__name__}: {e}>"
                print(f"  {attr}: {value}")

        print("\n  public attrs:")
        for name in sorted(n for n in dir(runner) if not n.startswith("_")):
            print(f"    - {name}")
    finally:
        close = getattr(runner, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


if __name__ == "__main__":
    args = _build_parser().parse_args()
    inspect(args.model_path, device=args.device)

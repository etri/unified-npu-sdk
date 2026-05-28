# examples/inspect_rbln_model.py
"""
RBLN 컴파일 결과(.rbln) 파일의 입출력 텐서 정보를 출력합니다.
rebel.Runtime 객체에서 사용 가능한 메타 정보를 best-effort로 덤프합니다.
"""
from pathlib import Path
import sys


def _resolve_repo_root() -> Path:
    ws_root = Path("/workspace/unified-sdk")
    if ws_root.is_dir():
        return ws_root
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _resolve_repo_root()
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import rebel
except ImportError:
    print("Error: 'rebel' (Rebellions SDK) not found. Install via Rebellions private index first.")
    sys.exit(1)


def inspect(model_path: str) -> None:
    p = Path(model_path)
    if not p.is_file():
        print(f"Error: file not found - {p}")
        return

    try:
        rt = rebel.Runtime(str(p))
    except Exception as e:
        print(f"Error: failed to load runtime ({type(e).__name__}): {e}")
        return

    print(f"\n== RBLN model: {p.name} ==")
    # rebel.Runtime API는 버전에 따라 노출 속성이 다르므로 best-effort 출력
    for attr in ("get_input_info", "get_output_info", "input_info", "output_info"):
        if hasattr(rt, attr):
            value = getattr(rt, attr)
            try:
                value = value() if callable(value) else value
            except Exception as e:
                value = f"<{type(e).__name__}: {e}>"
            print(f"  {attr}: {value}")

    # 일반 dir() 덤프 (디버깅 도움용)
    print("\n  public attrs:")
    for name in sorted(n for n in dir(rt) if not n.startswith("_")):
        print(f"    - {name}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "builds" / "resnet50.rbln")
    inspect(target)

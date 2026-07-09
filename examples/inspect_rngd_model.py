# examples/inspect_rngd_model.py
"""
RNGD 아티팩트(furiosa-llm ArtifactBuilder 결과) 또는 HF 모델 id 의 메타 정보를 출력합니다.
아티팩트 디렉터리면 구성 파일을 나열하고, HF 모델 id 면 런타임에서 로드된다는 안내를 출력합니다.
--load 를 주면 furiosa_llm.LLM 으로 실제 로드해 best-effort 속성 덤프를 시도합니다(무거움).
"""
import argparse
import json
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

DEFAULT_ENGINE = os.getenv("RNGD_MODEL", "furiosa-ai/Qwen2.5-0.5B-Instruct")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a FuriosaAI RNGD artifact or model id.")
    parser.add_argument("model_path", nargs="?", default=DEFAULT_ENGINE)
    parser.add_argument("--load", action="store_true", help="furiosa_llm.LLM 으로 실제 로드해 속성 덤프(무거움).")
    return parser


def _describe_artifact_dir(p: Path) -> None:
    print(f"\n== RNGD artifact dir: {p} ==")
    files = sorted(x.name for x in p.iterdir())
    print("  files:")
    for name in files:
        print(f"    - {name}")
    for cfg_name in ("config.json", "artifact.json", "model_metadata.json"):
        cfg = p / cfg_name
        if cfg.is_file():
            try:
                data = json.loads(cfg.read_text())
                keys = list(data.keys()) if isinstance(data, dict) else type(data).__name__
                print(f"  {cfg_name} keys: {keys}")
            except Exception as e:
                print(f"  {cfg_name}: <unreadable: {e}>")


def _load_and_dump(engine: str) -> None:
    try:
        from furiosa_llm import LLM
    except ImportError:
        print("Error: 'furiosa_llm' not found. Install furiosa-llm first (developer.furiosa.ai).")
        return
    try:
        llm = LLM.from_artifacts(engine) if Path(engine).is_dir() else LLM(engine)
    except Exception as e:
        print(f"Error: failed to load LLM ({type(e).__name__}): {e}")
        return
    print("\n  public attrs:")
    for name in sorted(n for n in dir(llm) if not n.startswith("_")):
        print(f"    - {name}")
    tok = getattr(llm, "tokenizer", None)
    if tok is not None:
        print("  tokenizer:", type(tok).__name__)


if __name__ == "__main__":
    args = _build_parser().parse_args()
    engine = str(args.model_path)
    p = Path(engine)

    if p.is_dir():
        _describe_artifact_dir(p)
    else:
        print(f"\n== RNGD model id: {engine} ==")
        print("  (HuggingFace model id — furiosa_llm.LLM 이 런타임에 로드합니다)")

    if args.load:
        _load_and_dump(engine)

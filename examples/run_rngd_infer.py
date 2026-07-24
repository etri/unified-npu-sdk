# examples/run_rngd_infer.py
import argparse
import timeit
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
    parser = argparse.ArgumentParser(description="Generate text with a FuriosaAI RNGD LLM (furiosa-llm).")
    parser.add_argument("--engine-path", default=DEFAULT_ENGINE,
                        help="HF 모델 id 또는 로컬 모델 경로.")
    parser.add_argument("--fxb-path", default=None,
                        help="선택 기능: 명시적으로 사용할 FXB 파일 경로. custom smoke 에서 사용.")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--devices", default=os.getenv("FURIOSA_DEVICES", None), help="예: 'npu:0'.")
    parser.add_argument("--chat", action="store_true",
                        help="LLM 토크나이저의 chat template 을 적용해 프롬프트를 감싼다.")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.fxb_path:
        fxb_path = Path(args.fxb_path).expanduser()
        if not fxb_path.is_file():
            raise SystemExit(
                f"Error: --fxb-path does not point to an existing FXB file: {fxb_path}\n"
                "Custom smoke requires a successful `fxb build` first."
            )

    from unified_sdk.types import RuntimeConfig
    from unified_sdk.runtime import create_runtime_LLM, generate_LLM, destroy_runtime_LLM

    cfg = RuntimeConfig(
        backend="rngd",
        engine_path=str(args.engine_path),
        fxb_path=str(args.fxb_path) if args.fxb_path else None,
        devices=args.devices,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_tokens=args.min_tokens,
    )

    rh = create_runtime_LLM(cfg)

    prompt = args.prompt
    if args.chat:
        # chat 모델은 tokenizer.apply_chat_template 로 감싸는 것이 정석 (참조 가이드 code-21).
        llm = rh.ctx.get("llm")
        tok = getattr(llm, "tokenizer", None)
        if tok is not None and hasattr(tok, "apply_chat_template"):
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": args.prompt}], tokenize=False
            )

    t0 = timeit.default_timer()
    text = generate_LLM(rh, prompt)
    dt = (timeit.default_timer() - t0) * 1000

    print("== RNGD generate ==")
    print("engine =", args.engine_path)
    print("fxb =", args.fxb_path)
    print("prompt =", args.prompt)
    print("---- output ----")
    print(text)
    print("----------------")
    print(f"latency_ms = {dt:.1f}")

    destroy_runtime_LLM(rh)

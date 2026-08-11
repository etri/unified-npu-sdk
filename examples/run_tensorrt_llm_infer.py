import argparse
from pathlib import Path
import sys
import os
import timeit
from typing import Any


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
    parser = argparse.ArgumentParser(description="Run TensorRT-LLM generate through Unified SDK LLM API.")
    parser.add_argument("--model-ref-or-path", dest="model_ref_or_path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--engine-path", dest="model_ref_or_path", help=argparse.SUPPRESS)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--prompt", default="What is the capital of South Korea?")
    parser.add_argument("--chat-template", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--iters", type=int, default=1)
    return parser


def _resolve_tokenizer_ref(model_ref_or_path: str, tokenizer_path: str | None) -> str | None:
    if tokenizer_path:
        return tokenizer_path

    p = Path(model_ref_or_path).expanduser()
    if p.exists():
        if p.is_dir():
            for marker in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
                if (p / marker).exists():
                    return str(p)
            return None
        return None

    return model_ref_or_path


def _format_prompt(prompt: str, tokenizer_ref: str | None, trust_remote_code: bool, chat_template: str) -> str:
    if chat_template == "off" or not tokenizer_ref:
        return prompt

    try:
        from transformers import AutoTokenizer
    except Exception:
        return prompt if chat_template == "auto" else prompt

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, trust_remote_code=trust_remote_code)
    except Exception:
        return prompt if chat_template == "auto" else prompt

    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt

    template = getattr(tokenizer, "chat_template", None)
    if chat_template == "auto" and not template:
        return prompt

    try:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(formatted, str) and formatted:
            return formatted
    except Exception:
        if chat_template == "on":
            raise
    return prompt


def _render_output(result: Any) -> str:
    if isinstance(result, str):
        return result
    outputs = getattr(result, "outputs", None)
    if outputs:
        text = getattr(outputs[0], "text", None)
        if isinstance(text, str):
            return text
        token_ids = getattr(outputs[0], "token_ids", None)
        if token_ids is not None:
            return f"<empty text; token_ids={token_ids}>"
        return str(outputs[0])
    if isinstance(result, (list, tuple)):
        if not result:
            return ""
        first = result[0]
        if isinstance(first, str):
            return first
        outputs = getattr(first, "outputs", None)
        if outputs:
            text = getattr(outputs[0], "text", None)
            if isinstance(text, str):
                return text
        text = getattr(first, "text", None)
        if isinstance(text, str):
            return text
        return str(first)
    return str(result)


if __name__ == "__main__":
    args = _build_parser().parse_args()

    try:
        from unified_sdk.frontends import resolve_tensorrt_llm_fetch_request
        from unified_sdk.frontends.types import TensorRTLLMFrontendFetchRequest
        from unified_sdk.options import TensorRTLLMRuntimeOptions
        from unified_sdk.types import LLMRuntimeConfig
        from unified_sdk.runtime import create_runtime_LLM, destroy_runtime_LLM, generate_LLM
    except ImportError:
        print("Error: 'unified_sdk' package not found. Install it first or run from the repository checkout.")
        sys.exit(1)

    prepared_fetch = resolve_tensorrt_llm_fetch_request(
        TensorRTLLMFrontendFetchRequest(model_ref=args.model_ref_or_path)
    )
    cfg = LLMRuntimeConfig(
        backend="tensorrt",
        model_ref_or_path=args.model_ref_or_path,
        prepared_fetch_input=prepared_fetch.prepared_input,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_tokens=args.min_tokens,
        backend_options=TensorRTLLMRuntimeOptions(
            tokenizer_path=args.tokenizer_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        ),
    )
    rh = create_runtime_LLM(cfg)
    tokenizer_ref = _resolve_tokenizer_ref(args.model_ref_or_path, args.tokenizer_path)
    formatted_prompt = _format_prompt(
        args.prompt,
        tokenizer_ref=tokenizer_ref,
        trust_remote_code=args.trust_remote_code,
        chat_template=args.chat_template,
    )

    latencies = []
    result = None
    try:
        for _ in range(args.iters):
            t0 = timeit.default_timer()
            result = generate_LLM(rh, formatted_prompt)
            t1 = timeit.default_timer()
            latencies.append((t1 - t0) * 1000.0)

        print("== TensorRT-LLM generate ==")
        print(f"repo_root = {REPO_ROOT}")
        print(f"model_ref_or_path = {args.model_ref_or_path}")
        print(f"tokenizer_ref = {tokenizer_ref}")
        print(f"runtime_mode = {rh.ctx.get('runtime_mode')}")
        print(f"runtime_entry_kind = {rh.ctx.get('runtime_entry_kind')}")
        print(f"runtime_may_trigger_vendor_build = {rh.ctx.get('runtime_may_trigger_vendor_build')}")
        print(f"prompt = {args.prompt}")
        print(f"formatted_prompt = {formatted_prompt}")
        print(f"response = {_render_output(result)}")
        print(f"latency_ms = {sum(latencies) / len(latencies):.3f}")
    finally:
        destroy_runtime_LLM(rh)

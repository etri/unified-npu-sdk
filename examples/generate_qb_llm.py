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
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "Llama-3.2-1B-Instruct"
DEFAULT_MODEL_ID = os.getenv("QB_LLM_MODEL_ID", "mobilint/Llama-3.2-1B-Instruct")
DEFAULT_PROMPT = "대한민국의 수도는 어디인가요?"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal QB LLM generate preview helper. "
            "This wraps the vendor-provided Transformers trust_remote_code path "
            "to answer a simple prompt with a precompiled Mobilint LLM MXQ."
        )
    )
    parser.add_argument(
        "--model-ref",
        default=None,
        help=(
            "로컬 snapshot 디렉터리 또는 Hugging Face model id. "
            f"기본은 local={DEFAULT_MODEL_DIR} 가 있으면 그쪽, 없으면 {DEFAULT_MODEL_ID!r}."
        ),
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="chat template 대신 raw prompt 토큰화를 시도합니다.",
    )
    return parser


def _resolve_model_ref(model_ref: str | None) -> str:
    if model_ref:
        return model_ref
    if DEFAULT_MODEL_DIR.is_dir():
        return str(DEFAULT_MODEL_DIR)
    return DEFAULT_MODEL_ID


def _build_inputs(tokenizer, prompt: str, *, raw: bool):
    if raw:
        return tokenizer(prompt, return_tensors="pt")

    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    return tokenizer(prompt, return_tensors="pt")


def _extract_generated_text(tokenizer, inputs, outputs) -> str:
    input_ids = inputs.get("input_ids")
    if input_ids is not None and hasattr(input_ids, "shape") and len(input_ids.shape) >= 2:
        prompt_len = int(input_ids.shape[-1])
        generated = outputs[0][prompt_len:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


if __name__ == "__main__":
    args = _build_parser().parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise SystemExit(
            "Error: transformers is required. Rebuild the qb-only image first. "
            f"({type(exc).__name__}: {exc})"
        )

    model_ref = _resolve_model_ref(args.model_ref)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_ref, trust_remote_code=True)
    except Exception as exc:
        raise SystemExit(
            "Failed to initialize the Mobilint LLM generate preview helper. "
            "If you are using a local snapshot, retry after "
            "`prepare_qb_transformer_model.py --full-snapshot` so that remote-code files "
            "and any model-side auxiliary assets are present. "
            "This helper is a minimal wrapper around a vendor-provided path and may depend on "
            "vendor-side support details. "
            f"({type(exc).__name__}: {exc})"
        )

    try:
        inputs = _build_inputs(tokenizer, args.prompt, raw=args.raw)
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        text = _extract_generated_text(tokenizer, inputs, outputs)
    except Exception as exc:
        raise SystemExit(
            "QB generate preview failed while running the vendor-provided generate path. "
            "This helper is intentionally minimal and vendor-dependent; future updates may improve it "
            "when more official support or guidance becomes available. "
            f"({type(exc).__name__}: {exc})"
        )

    print("== QB LLM generate preview ==")
    print(f"(repo_root={REPO_ROOT})")
    print(f"(model_ref={model_ref})")
    print(f"(prompt={args.prompt!r})")
    print(f"(max_new_tokens={args.max_new_tokens})")
    print("")
    print(text)

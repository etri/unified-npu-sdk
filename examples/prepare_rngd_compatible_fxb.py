import argparse
from pathlib import Path
import subprocess
import sys


DEFAULT_MODEL = "Qwen/Qwen3-8B-FP8"
DEFAULT_BUNDLE_REPO = "furiosa-ai/Qwen3-8B-FP8"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download a compatible prebuilt FXB and print the recommended cached path for "
            "custom local-model runtime smoke."
        )
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Target upstream/raw Hugging Face model repo id to check compatibility against.",
    )
    parser.add_argument(
        "--bundle-repo",
        default=DEFAULT_BUNDLE_REPO,
        help="FuriosaAI Hugging Face repo id that publishes a compatible prebuilt FXB bundle.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional FXB cache directory override.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip `fxb download` and only run compatibility check.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download bundle even if it already exists in the cache.",
    )
    return parser


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            "Error: `fxb` command not found. Install furiosa-llm first "
            "(see https://developer.furiosa.ai/latest/en/furiosa_llm/fxb.html)."
        ) from exc


def _build_cache_args(cache_dir: str | None) -> list[str]:
    if not cache_dir:
        return []
    return ["--cache-dir", cache_dir]


def _extract_recommended_path(output: str) -> str | None:
    for line in output.splitlines():
        if line.startswith("Recommended:"):
            return line.split("Recommended:", 1)[1].strip()
    return None


if __name__ == "__main__":
    args = _build_parser().parse_args()

    cache_args = _build_cache_args(args.cache_dir)

    if not args.skip_download:
        download_cmd = ["fxb", "download", args.bundle_repo, *cache_args]
        if args.force:
            download_cmd.append("--force")
        download = _run(download_cmd)
        if download.returncode != 0:
            detail = (download.stderr or download.stdout or "").strip()
            raise SystemExit(f"Error: `fxb download` failed: {detail}")

    check_cmd = ["fxb", "check", args.model, *cache_args]
    check = _run(check_cmd)
    if check.returncode != 0:
        detail = (check.stderr or check.stdout or "").strip()
        raise SystemExit(f"Error: `fxb check` failed: {detail}")

    recommended = _extract_recommended_path(check.stdout or "")
    print("Complete!")
    print(f"(model={args.model})")
    print(f"(bundle_repo={args.bundle_repo})")
    if args.cache_dir:
        print(f"(cache_dir={Path(args.cache_dir).expanduser().resolve()})")
    if recommended:
        print(f"(recommended_fxb={recommended})")
    else:
        print("(recommended_fxb=not found in `fxb check` output; inspect `fxb cache ls` manually)")

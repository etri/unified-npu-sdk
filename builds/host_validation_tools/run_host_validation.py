from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_LOG_ROOT = SCRIPT_DIR / "logs"


def _run_to_log(command: list[str], log_path: Path) -> int:
    print(f"[run] {' '.join(command)}")
    print(f"[log] {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {' '.join(command)}\n\n")
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log.write(process.stdout)
    print(f"[exit] {process.returncode}\n")
    return process.returncode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TensorRT host-native validation scripts.")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-resnet", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = args.log_root / stamp

    steps: list[tuple[str, list[str]]] = [
        ("host_env", [sys.executable, str(SCRIPT_DIR / "collect_env.py")]),
    ]
    if not args.skip_smoke:
        steps.append(("smoke_conv_compile", [sys.executable, str(SCRIPT_DIR / "smoke_conv_compile.py")]))
    if not args.skip_resnet:
        steps.append(("resnet50_compile", [sys.executable, str(SCRIPT_DIR / "resnet50_compile.py")]))
    if not args.skip_infer:
        steps.append(("resnet50_infer", [sys.executable, str(SCRIPT_DIR / "resnet50_infer.py")]))

    failures = []
    for name, command in steps:
        code = _run_to_log(command, log_dir / f"{name}.log")
        if code != 0:
            failures.append((name, code))

    print(f"Logs written to: {log_dir}")
    if failures:
        print("Failed steps:")
        for name, code in failures:
            print(f"- {name}: exit_code={code}")
        return 1

    print("All host validation steps passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

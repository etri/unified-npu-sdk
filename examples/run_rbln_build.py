# examples/run_rbln_build.py
import argparse
from pathlib import Path
import sys
import os

def _is_repo_root(path: Path) -> bool:
    return (path / "src" / "unified_sdk").is_dir() and (path / "examples").is_dir()


def _resolve_repo_root() -> Path:
    """
    기준:
      1) 환경 변수 UNIFIED_SDK_REPO_ROOT 가 있으면 우선 사용
      2) 현재 작업 디렉터리가 repo root 구조면 그걸 사용
      3) 현재 파일 위치(.../examples/run_rbln_build.py) 기준으로 checkout root 추론
      4) 마지막 fallback 으로 알려진 컨테이너 경로를 확인
    """
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

try:
    from unified_sdk.types import BuildConfig
    from unified_sdk.build.api import build_unified
except ImportError:
    print("Error: 'unified_sdk' package not found. Install it first or run from the repository checkout.")
    sys.exit(1)
EXAMPLES_DIR = REPO_ROOT / "examples"
MODELS_DIR = REPO_ROOT / "models"
BUILDS_DIR = REPO_ROOT / "builds"


def _parse_shape(value: str) -> tuple[int, ...]:
    parts = value.replace("x", ",").split(",")
    try:
        shape = tuple(int(part.strip()) for part in parts if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape: {value!r}") from exc
    if not shape or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError(f"shape must contain positive integers: {value!r}")
    return shape


def _parse_bucketing_shapes(value: str | None) -> list[tuple[int, ...]] | None:
    if not value:
        return None
    return [_parse_shape(item) for item in value.split(";") if item.strip()]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile torchvision ResNet50 to an RBLN model.")
    parser.add_argument("--weights", type=Path, default=None, help="Path to resnet50 .pth/.pt weights.")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="Directory used to find weights.")
    parser.add_argument("--out-dir", type=Path, default=BUILDS_DIR, help="Directory for compiled .rbln output.")
    parser.add_argument("--model-name", default="resnet50", help="Output model name without extension.")
    parser.add_argument(
        "--require-weights",
        action="store_true",
        help="Fail if no local ResNet50 weights are found. Default smoke mode uses random weights.",
    )
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--input-shape", type=_parse_shape, default=(1, 3, 224, 224))
    parser.add_argument(
        "--bucketing-shapes",
        default=None,
        help="Semicolon-separated input shapes, e.g. '1,3,224,224;4,3,224,224'.",
    )
    parser.add_argument(
        "--model-trace-method",
        choices=("export", "export_strict", "jittrace"),
        default=None,
        help="Optional RBLN trace method passed to compile_from_torch.",
    )
    parser.add_argument("--npu", default=os.getenv("RBLN_NPU_NAME", "RBLN-CA22"))
    return parser


def _check_repo_layout(models_dir: Path, out_dir: Path):
    missing = []
    if not EXAMPLES_DIR.is_dir():
        missing.append(str(EXAMPLES_DIR))

    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if missing:
        raise FileNotFoundError(
            "필수 폴더가 없습니다:\n"
            + "\n".join(f"- {p}" for p in missing)
            + f"\n\n(현재 기준 REPO_ROOT = {REPO_ROOT})"
        )


def _find_weights(models_dir: Path) -> Path | None:
    candidates = sorted(models_dir.glob("resnet50*.pth")) + sorted(models_dir.glob("resnet50*.pt"))
    if not candidates:
        return None
    return candidates[0]


def _load_state_dict(path: Path, torch_module) -> dict:
    obj = torch_module.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        elif "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            obj = obj["model_state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"가중치 파일 형식을 해석할 수 없습니다: {path}")

    cleaned = {}
    for k, v in obj.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


if __name__ == "__main__":
    args = _build_parser().parse_args()

    try:
        import torch
        from torchvision.models import resnet50
    except ImportError:
        print("Error: 'torch' and 'torchvision' are required for the RBLN build example.")
        sys.exit(1)

    models_dir = args.models_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()

    _check_repo_layout(models_dir, out_dir)

    weights_path = args.weights.expanduser().resolve() if args.weights else _find_weights(models_dir)
    if args.require_weights and weights_path is None:
        raise FileNotFoundError(
            f"{models_dir} 에서 resnet50 가중치 파일을 찾지 못했습니다.\n"
            f"예) {models_dir/'resnet50.pth'} 또는 {models_dir/'resnet50_state_dict.pth'}"
        )
    if weights_path is not None and not weights_path.is_file():
        raise FileNotFoundError(f"weights file not found: {weights_path}")

    # 기본 smoke는 외부 다운로드 없이 ResNet50 random weight 모델로 컴파일 경로를 검증합니다.
    model = resnet50(weights=None)
    if weights_path is not None:
        sd = _load_state_dict(weights_path, torch)
        model.load_state_dict(sd, strict=False)
    model.eval()

    cfg = BuildConfig(
        backend="rbln",
        model_or_path=model,
        out_dir=str(out_dir),
        model_name=args.model_name,
        precision=args.precision,
        input_name=args.input_name,
        input_shape=args.input_shape,
        bucketing_shapes=_parse_bucketing_shapes(args.bucketing_shapes),
        extra={
            key: value
            for key, value in {
                "npu": args.npu,
                "model_trace_method": args.model_trace_method,
            }.items()
            if value
        },
    )

    result = build_unified(cfg)
    print("Complete!", result.compiled_model_path)
    print(f"(repo_root={REPO_ROOT})")
    print(f"(weights={weights_path if weights_path else 'random-init smoke'})")

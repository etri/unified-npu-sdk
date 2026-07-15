# examples/run_warboy_build.py
import argparse
from pathlib import Path
import sys
import os
import re

def _is_repo_root(path: Path) -> bool:
    return (path / "src" / "unified_sdk").is_dir() and (path / "examples").is_dir()


def _resolve_repo_root() -> Path:
    """
    기준:
      1) 환경 변수 UNIFIED_SDK_REPO_ROOT 가 있으면 우선 사용
      2) 현재 작업 디렉터리가 repo root 구조면 그걸 사용
      3) 현재 파일 위치(.../examples/run_warboy_build.py) 기준으로 checkout root 추론
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a FuriosaAI Warboy .enf model. "
        "기본은 Furiosa model zoo ENF fetch 또는 사전 컴파일된 .enf 확보(fetch), "
        "--from-onnx(quantized ONNX) 지정 시 furiosa-compiler 컴파일(compile hook). "
        ".pth/.pt 가중치 파일은 직접 입력으로 지원하지 않습니다."
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="fetch 모드에서 .enf 를 찾을 디렉터리.")
    parser.add_argument("--out-dir", type=Path, default=BUILDS_DIR, help="결과 .enf 출력 디렉터리.")
    parser.add_argument("--model-name", default="resnet50", help="확장자 없는 출력 모델 이름.")
    parser.add_argument("--enf", type=Path, default=None, help="이미 컴파일된 .enf 를 직접 사용(fetch/provided).")
    parser.add_argument("--from-onnx", type=Path, default=None, help="quantized ONNX 를 furiosa-compiler 로 .enf 컴파일(compile hook).")
    parser.add_argument("--target-npu", choices=("warboy", "warboy-2pe"), default="warboy-2pe")
    parser.add_argument("--require-enf", action="store_true", help="fetch 모드에서 .enf 를 못 찾으면 실패 처리.")
    parser.add_argument(
        "--list-model-zoo",
        action="store_true",
        help="설치된 furiosa.models.vision model zoo 에서 사용 가능한 모델 이름 후보를 출력하고 종료합니다.",
    )
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--input-shape", type=_parse_shape, default=(1, 3, 224, 224))
    return parser


def _find_enf(models_dir: Path, model_name: str) -> Path | None:
    normalized = _normalize_model_name(model_name)
    for candidate in sorted(models_dir.glob("*.enf")):
        if _normalize_model_name(candidate.stem) == normalized:
            return candidate
    return None


def _normalize_model_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _list_model_zoo_targets() -> list[str]:
    try:
        from furiosa.models import vision
    except Exception:
        return []

    declared = getattr(vision, "__all__", None)
    if isinstance(declared, (list, tuple)):
        return sorted({str(name) for name in declared if isinstance(name, str) and not name.startswith("_")})

    return sorted({name for name in dir(vision) if not name.startswith("_")})


def _resolve_model_zoo_target(model_name: str) -> str | None:
    normalized = _normalize_model_name(model_name)
    for candidate in _list_model_zoo_targets():
        if _normalize_model_name(candidate) == normalized:
            return candidate
    return None


def _fetch_model_zoo_enf(model_name: str, target_npu: str, models_dir: Path) -> Path | None:
    try:
        from furiosa.models import vision
    except Exception:
        return None

    resolved = _resolve_model_zoo_target(model_name)
    if resolved is None:
        return None

    try:
        model_cls = getattr(vision, resolved, None)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import Furiosa model zoo target {resolved!r}. "
            "Some vision models may require optional runtime dependencies in the current image."
        ) from exc
    if not callable(model_cls):
        return None

    try:
        model = model_cls()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to construct Furiosa model zoo target {resolved!r}. "
            "Check optional dependencies for this model family in the current image."
        ) from exc
    num_pe = 1 if target_npu == "warboy" else 2
    model_source = model.model_source(num_pe=num_pe)

    out_path = (models_dir / model_name).with_suffix(".enf")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model_source, (bytes, bytearray)):
        out_path.write_bytes(model_source)
    else:
        source_path = Path(str(model_source)).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"model zoo ENF source not found: {source_path}")
        out_path.write_bytes(source_path.read_bytes())
    return out_path


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.list_model_zoo:
        items = _list_model_zoo_targets()
        if not items:
            print("furiosa.models.vision model zoo 목록을 찾지 못했습니다.")
            sys.exit(1)
        print("== Available Furiosa model zoo targets ==")
        for name in items:
            print(name)
        sys.exit(0)

    models_dir = args.models_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    extra: dict = {"target_npu": args.target_npu, "target_ir": "enf"}

    # 우선순위: --from-onnx(compile) > --enf(provided) > models/ 자동탐지(fetch) > model zoo ENF fetch
    if args.from_onnx is not None:
        onnx_path = args.from_onnx.expanduser().resolve()
        if not onnx_path.is_file():
            raise FileNotFoundError(f"quantized ONNX not found: {onnx_path}")
        model_or_path: str = str(onnx_path)
        source_desc = f"furiosa-compiler from quantized ONNX: {onnx_path}"
    else:
        fetched_from_model_zoo = False
        enf = args.enf.expanduser().resolve() if args.enf else _find_enf(models_dir, args.model_name)
        if enf is None:
            enf = _fetch_model_zoo_enf(args.model_name, args.target_npu, models_dir)
            fetched_from_model_zoo = enf is not None
        if enf is None:
            msg = (
                f"{models_dir} 에서 {args.model_name}*.enf 를 찾지 못했고, "
                "Furiosa model zoo 에서도 대응 ENF 를 확보하지 못했습니다.\n"
                "사전 컴파일된 .enf 를 --enf 로 지정하거나, --from-onnx <quantized.onnx> 로 컴파일하세요.\n"
                ".pth/.pt 가중치 파일은 직접 입력으로 지원하지 않습니다."
            )
            if args.require_enf:
                raise FileNotFoundError(msg)
            print("[WARN] " + msg)
            sys.exit(1)
        model_or_path = str(enf)
        if fetched_from_model_zoo:
            source_desc = f"standard fetch from Furiosa model zoo ENF: {enf}"
        else:
            source_desc = f"provided .enf: {enf}"

    cfg = BuildConfig(
        backend="warboy",
        model_or_path=model_or_path,
        out_dir=str(out_dir),
        model_name=args.model_name,
        precision="int8",
        input_name=args.input_name,
        input_shape=args.input_shape,
        extra=extra,
    )

    result = build_unified(cfg)
    print("Complete!", result.compiled_model_path)
    print(f"(repo_root={REPO_ROOT})")
    print(f"(source={source_desc})")

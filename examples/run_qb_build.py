# examples/run_qb_build.py
import argparse
from pathlib import Path
import sys
import os
from typing import Any

def _is_repo_root(path: Path) -> bool:
    return (path / "src" / "unified_sdk").is_dir() and (path / "examples").is_dir()


def _resolve_repo_root() -> Path:
    """
    기준:
      1) 환경 변수 UNIFIED_SDK_REPO_ROOT 가 있으면 우선 사용
      2) 현재 작업 디렉터리가 repo root 구조면 그걸 사용
      3) 현재 파일 위치(.../examples/run_qb_build.py) 기준으로 checkout root 추론
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
        description="Build a Mobilint ARISE (QB) .mxq model. "
        "기본은 표준 model zoo / 로컬 .mxq 확보(fetch), "
        "--from-onnx 또는 --from-pth 지정 시 Mobilint compiler Python API(qubee/qbcompiler)로 컴파일(compile hook)."
    )
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="fetch 모드에서 먼저 탐색할 .mxq 디렉터리.")
    parser.add_argument("--out-dir", type=Path, default=BUILDS_DIR, help="결과 .mxq 출력 디렉터리.")
    parser.add_argument("--model-name", default="resnet50", help="확장자 없는 출력 모델 이름.")
    parser.add_argument("--mxq", type=Path, default=None, help="이미 컴파일된 .mxq 를 직접 사용(fetch/provided).")
    parser.add_argument("--from-onnx", type=Path, default=None, help="이 ONNX 를 Mobilint compiler Python API(qubee/qbcompiler)로 .mxq 컴파일(compile hook).")
    parser.add_argument("--from-pth", type=Path, default=None, help="이 .pth/.pt weights를 ONNX로 export한 뒤 .mxq 컴파일(현재 resnet50 예제 지원).")
    parser.add_argument("--export-onnx-path", type=Path, default=None, help="--from-pth 사용 시 생성할 중간 ONNX 경로 (기본: models/<model-name>.onnx).")
    parser.add_argument("--calib", type=Path, default=None, help="Mobilint compiler calibration 데이터셋 메타 파일(.txt/.json).")
    parser.add_argument(
        "--quantize-method",
        choices=("percentile", "maxpercentile", "max", "kl"),
        default="percentile",
    )
    parser.add_argument(
        "--use-random-calib",
        action="store_true",
        help="calibration 데이터 없이 random calib 로 컴파일(smoke).",
    )
    parser.add_argument(
        "--require-mxq",
        action="store_true",
        help="fetch 모드에서 .mxq 를 못 찾으면 실패 처리.",
    )
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--input-shape", type=_parse_shape, default=(1, 3, 224, 224))
    parser.add_argument("--core-mode", default=os.getenv("MBLT_CORE_MODE", "global8"))
    parser.add_argument("--product", default=os.getenv("MBLT_PRODUCT", "aries"))
    return parser


def _unwrap_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "weights"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                payload = nested
                break
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    return payload


def _prepare_module_from_pth(weights_path: Path, model_name: str):
    try:
        import torch
        from torchvision import models as tv_models
    except ImportError as exc:
        raise RuntimeError(
            "torch and torchvision are required to compile from .pth/.pt weights."
        ) from exc

    normalized = model_name.lower()
    if normalized != "resnet50":
        raise ValueError(
            "--from-pth currently supports only --model-name resnet50. "
            "Use --from-onnx for other architectures."
        )

    checkpoint = torch.load(str(weights_path), map_location="cpu")
    state_dict = _unwrap_state_dict(checkpoint)
    model = tv_models.resnet50(weights=None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Failed to load resnet50 weights cleanly from checkpoint. "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    model.eval()
    return model


def _export_onnx_from_pth(weights_path: Path, export_onnx_path: Path, model_name: str, input_name: str, input_shape: tuple[int, ...]) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to export ONNX from .pth/.pt weights.") from exc

    model = _prepare_module_from_pth(weights_path, model_name)
    export_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(input_shape, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(export_onnx_path),
        input_names=[input_name],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True,
    )
    if not export_onnx_path.is_file():
        raise RuntimeError(f"ONNX export did not produce a file: {export_onnx_path}")
    return export_onnx_path


def _find_mxq(models_dir: Path, model_name: str) -> Path | None:
    candidates = sorted(models_dir.glob(f"{model_name}*.mxq")) + sorted(models_dir.glob("*.mxq"))
    return candidates[0] if candidates else None


def _find_model_zoo_mxq(model_name: str, product: str, core_mode: str) -> Path | None:
    home = Path.home()
    normalized = model_name.lower()
    zoo_root = home / ".mblt_model_zoo" / "vision" / product / core_mode

    explicit_candidates = [
        zoo_root / f"{normalized}_IMAGENET1K_V2.mxq",
        zoo_root / f"{normalized}_DEFAULT.mxq",
    ]
    for candidate in explicit_candidates:
        if candidate.is_file():
            return candidate

    glob_candidates = sorted(zoo_root.glob(f"{normalized}*.mxq"))
    return glob_candidates[0] if glob_candidates else None


if __name__ == "__main__":
    args = _build_parser().parse_args()

    models_dir = args.models_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    extra: dict = {"quantize_method": args.quantize_method, "core_mode": args.core_mode}
    if args.use_random_calib:
        extra["use_random_calib"] = True

    # 우선순위:
    #   1) --from-pth    : local weights -> ONNX export -> compile
    #   2) --from-onnx   : local/custom ONNX -> compile
    #   3) --mxq         : custom/local precompiled .mxq fetch
    #   4) models/       : local/custom fetch
    #   5) ~/.mblt_model_zoo : standard fetch
    if args.from_pth is not None:
        weights_path = args.from_pth.expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"PTH/PT weights not found: {weights_path}")
        export_onnx_path = (
            args.export_onnx_path.expanduser().resolve()
            if args.export_onnx_path is not None
            else (models_dir / f"{args.model_name}.onnx").resolve()
        )
        onnx_path = _export_onnx_from_pth(
            weights_path=weights_path,
            export_onnx_path=export_onnx_path,
            model_name=args.model_name,
            input_name=args.input_name,
            input_shape=args.input_shape,
        )
        model_or_path = str(onnx_path)
        calib = str(args.calib.expanduser().resolve()) if args.calib else None
        source_desc = f"local weights -> ONNX export -> compiler Python API compile: {weights_path} -> {onnx_path}"
    elif args.from_onnx is not None:
        onnx_path = args.from_onnx.expanduser().resolve()
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")
        model_or_path: str = str(onnx_path)
        calib = str(args.calib.expanduser().resolve()) if args.calib else None
        source_desc = f"local/custom ONNX -> compiler Python API compile: {onnx_path}"
    else:
        mxq = args.mxq.expanduser().resolve() if args.mxq else _find_mxq(models_dir, args.model_name)
        source_desc = ""
        if mxq is None:
            mxq = _find_model_zoo_mxq(args.model_name, args.product, args.core_mode)
            if mxq is not None:
                source_desc = f"standard fetch from official model zoo: {mxq}"
        if mxq is None:
            msg = (
                f"{models_dir} 또는 ~/.mblt_model_zoo/vision/{args.product}/{args.core_mode} 에서 "
                f"{args.model_name}*.mxq 를 찾지 못했습니다.\n"
                "표준 fetch는 ~/.mblt_model_zoo 의 .mxq 를 사용합니다.\n"
                "custom fetch는 --mxq <mxq> 로 로컬 경로를 지정하세요.\n"
                "custom compile은 --from-onnx <onnx> 또는 --from-pth <weights> 로 수행하세요."
            )
            if args.require_mxq:
                raise FileNotFoundError(msg)
            print("[WARN] " + msg)
            sys.exit(1)
        model_or_path = str(mxq)
        calib = None
        if not source_desc:
            source_desc = f"custom/local fetch from provided .mxq: {mxq}"

    cfg = BuildConfig(
        backend="qb",
        model_or_path=model_or_path,
        out_dir=str(out_dir),
        model_name=args.model_name,
        precision="int8",
        input_name=args.input_name,
        input_shape=args.input_shape,
        calib_data_path=calib,
        extra=extra,
    )

    result = build_unified(cfg)
    print("Complete!", result.compiled_model_path)
    print(f"(repo_root={REPO_ROOT})")
    print(f"(source={source_desc})")

import argparse
import itertools
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
MODELS_DIR = REPO_ROOT / "models"
TESTS_DIR = REPO_ROOT / "tests"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
        description="Prepare a quantized ONNX for FuriosaAI Warboy from torchvision ResNet50 weights."
    )
    parser.add_argument("--weights", type=Path, default=None, help="Path to resnet50 .pth/.pt weights.")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="Directory used to find weights and write ONNX outputs.")
    parser.add_argument("--model-name", default="resnet50", help="Base model name used for input/output file names.")
    parser.add_argument("--f32-onnx", type=Path, default=None, help="Optional f32 ONNX output path.")
    parser.add_argument("--quant-onnx", type=Path, default=None, help="Optional quantized ONNX output path.")
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--output-name", default="output")
    parser.add_argument("--input-shape", type=_parse_shape, default=(1, 3, 224, 224))
    parser.add_argument("--calib-dir", type=Path, default=None, help="Directory of calibration images.")
    parser.add_argument("--calib-image", type=Path, default=TESTS_DIR / "input.jpg", help="Single fallback calibration image.")
    parser.add_argument("--calib-iters", type=int, default=8, help="Number of calibration samples to collect.")
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow random-init ResNet50 when no .pth/.pt weights are found.",
    )
    return parser


def _find_weights(models_dir: Path, model_name: str) -> Path | None:
    candidates = sorted(models_dir.glob(f"{model_name}*.pth")) + sorted(models_dir.glob(f"{model_name}*.pt"))
    return candidates[0] if candidates else None


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
    for key, value in obj.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("model."):
            new_key = new_key[len("model.") :]
        cleaned[new_key] = value
    return cleaned


def _collect_image_candidates(calib_dir: Path | None, calib_image: Path) -> list[Path]:
    image_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    candidates: list[Path] = []
    if calib_dir and calib_dir.is_dir():
        for pattern in image_exts:
            candidates.extend(sorted(calib_dir.glob(pattern)))
    if calib_image.is_file():
        candidates.append(calib_image)

    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


if __name__ == "__main__":
    args = _build_parser().parse_args()

    try:
        import numpy as np
        import onnx
        import torch
        from torchvision.io.image import read_image
        from torchvision import transforms
        from torchvision.models import resnet50
        from furiosa.quantizer import quantize, Calibrator, CalibrationMethod
    except ImportError as exc:
        print(f"Error: missing dependency - {exc}")
        print("Need: torch, torchvision, onnx, furiosa-sdk[quantizer]")
        sys.exit(1)

    models_dir = args.models_dir.expanduser().resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    weights_path = args.weights.expanduser().resolve() if args.weights else _find_weights(models_dir, args.model_name)
    if weights_path is None and not args.allow_random_init:
        raise FileNotFoundError(
            f"{models_dir} 에서 {args.model_name}*.pth 또는 {args.model_name}*.pt 를 찾지 못했습니다.\n"
            "예) models/resnet50.pth\n"
            "가중치 없이 진행하려면 --allow-random-init 옵션을 사용하세요."
        )
    if weights_path is not None and not weights_path.is_file():
        raise FileNotFoundError(f"weights file not found: {weights_path}")

    f32_onnx = (
        args.f32_onnx.expanduser().resolve()
        if args.f32_onnx
        else (models_dir / f"{args.model_name}.onnx").resolve()
    )
    quant_onnx = (
        args.quant_onnx.expanduser().resolve()
        if args.quant_onnx
        else (models_dir / f"{args.model_name}_quantized.onnx").resolve()
    )

    print("== Warboy Quantized ONNX Prepare ==")
    print(f"(repo_root={REPO_ROOT})")
    print(f"(models_dir={models_dir})")

    model = resnet50(weights=None)
    if weights_path is not None:
        state_dict = _load_state_dict(weights_path, torch)
        model.load_state_dict(state_dict, strict=False)
        print(f"(weights={weights_path})")
    else:
        print("(weights=random-init smoke)")
    model.eval()

    dummy = torch.zeros(args.input_shape, dtype=torch.float32)
    f32_onnx.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(f32_onnx),
        input_names=[args.input_name],
        output_names=[args.output_name],
        opset_version=13,
    )
    print("onnx(f32) =", f32_onnx)

    preprocess = transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(args.input_shape[-1]),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    calib_dir = args.calib_dir.expanduser().resolve() if args.calib_dir else None
    calib_image = args.calib_image.expanduser().resolve()
    image_candidates = _collect_image_candidates(calib_dir, calib_image)

    f32 = onnx.load(str(f32_onnx))
    method = getattr(CalibrationMethod, "MIN_MAX_ASYM", None) or list(CalibrationMethod)[0]
    calibrator = Calibrator(f32, method)

    if image_candidates:
        print(f"(calibration=images x{args.calib_iters}, source_count={len(image_candidates)})")
        for image_path in itertools.islice(itertools.cycle(image_candidates), args.calib_iters):
            image = read_image(str(image_path))
            sample = preprocess(image).unsqueeze(0).numpy().astype(np.float32)
            calibrator.collect_data([[sample]])
    else:
        print(f"(calibration=synthetic random x{args.calib_iters})")
        for _ in range(args.calib_iters):
            sample = np.random.rand(*args.input_shape).astype(np.float32)
            calibrator.collect_data([[sample]])

    ranges = calibrator.compute_range()
    quantized = quantize(f32, ranges)
    quant_onnx.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(quantized, (bytes, bytearray)):
        quant_onnx.write_bytes(quantized)
    else:
        onnx.save(quantized, str(quant_onnx))
    print("onnx(quant) =", quant_onnx)

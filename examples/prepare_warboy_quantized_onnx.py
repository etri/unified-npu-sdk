import argparse
import itertools
from pathlib import Path
import sys
import os


class _Bottleneck:
    expansion = 4

    def __init__(self, nn_module, inplanes: int, planes: int, stride: int = 1, downsample=None) -> None:
        self.nn = nn_module
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.downsample = downsample

    def build(self):
        nn = self.nn

        class BottleneckModule(nn.Module):
            expansion = 4

            def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(
                    planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
                )
                self.bn2 = nn.BatchNorm2d(planes)
                self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
                self.bn3 = nn.BatchNorm2d(planes * self.expansion)
                self.relu = nn.ReLU(inplace=True)
                self.downsample = downsample
                self.stride = stride

            def forward(self, x):
                identity = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)

                out = self.conv3(out)
                out = self.bn3(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
                out = self.relu(out)
                return out

        return BottleneckModule(self.inplanes, self.planes, self.stride, self.downsample)


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
        description="Prepare a quantized ONNX for FuriosaAI Warboy from ResNet50 .pth/.pt weights."
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
    parser.add_argument("--calib-image", type=Path, default=TESTS_DIR / "input.jpg", help="Single calibration image.")
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


def _make_resnet50(torch_module):
    nn = torch_module.nn

    class ResNet(nn.Module):
        def __init__(self, block_factory, layers: list[int], num_classes: int = 1000) -> None:
            super().__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block_factory, 64, layers[0])
            self.layer2 = self._make_layer(block_factory, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block_factory, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block_factory, 512, layers[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * 4, num_classes)

        def _make_layer(self, block_factory, planes: int, blocks: int, stride: int = 1):
            downsample = None
            if stride != 1 or self.inplanes != planes * 4:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * 4),
                )

            layers = [block_factory(nn, self.inplanes, planes, stride, downsample).build()]
            self.inplanes = planes * 4
            for _ in range(1, blocks):
                layers.append(block_factory(nn, self.inplanes, planes).build())
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch_module.flatten(x, 1)
            x = self.fc(x)
            return x

    return ResNet(_Bottleneck, [3, 4, 6, 3])


def _center_crop_resize_pil(image, size: int):
    width, height = image.size
    scale = 256 / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)))
    left = max((resized.width - size) // 2, 0)
    top = max((resized.height - size) // 2, 0)
    return resized.crop((left, top, left + size, top + size))


def _load_calibration_sample(image_path: Path, input_shape: tuple[int, ...], np_module):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    cropped = _center_crop_resize_pil(image, input_shape[-1])
    array = np_module.asarray(cropped, dtype=np_module.float32) / 255.0
    array = array.transpose(2, 0, 1)
    mean = np_module.asarray(IMAGENET_MEAN, dtype=np_module.float32)[:, None, None]
    std = np_module.asarray(IMAGENET_STD, dtype=np_module.float32)[:, None, None]
    normalized = (array - mean) / std
    return normalized[None, ...].astype(np_module.float32)


def _load_and_infer_onnx(onnx_module, onnx_path: Path):
    model = onnx_module.load(str(onnx_path))
    try:
        from onnx import shape_inference

        inferred = shape_inference.infer_shapes(model)
        onnx_module.save(inferred, str(onnx_path))
        return inferred
    except Exception as exc:
        print(f"[WARN] ONNX shape inference skipped: {exc}")
        return model


if __name__ == "__main__":
    args = _build_parser().parse_args()

    try:
        import numpy as np
        import onnx
        import torch
        from furiosa.quantizer import quantize, Calibrator, CalibrationMethod
        from PIL import Image  # noqa: F401
    except ImportError as exc:
        print(f"Error: missing dependency - {exc}")
        print("Need: torch, onnx, pillow, furiosa-sdk[quantizer]")
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

    model = _make_resnet50(torch)
    if weights_path is not None:
        state_dict = _load_state_dict(weights_path, torch)
        load_result = model.load_state_dict(state_dict, strict=False)
        print(f"(weights={weights_path})")
        print(
            f"(load_state_dict: missing={len(load_result.missing_keys)}, "
            f"unexpected={len(load_result.unexpected_keys)})"
        )
        if load_result.missing_keys:
            print(f"[WARN] missing keys sample: {load_result.missing_keys[:5]}")
        if load_result.unexpected_keys:
            print(f"[WARN] unexpected keys sample: {load_result.unexpected_keys[:5]}")
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

    calib_dir = args.calib_dir.expanduser().resolve() if args.calib_dir else None
    calib_image = args.calib_image.expanduser().resolve()
    image_candidates = _collect_image_candidates(calib_dir, calib_image)

    f32 = _load_and_infer_onnx(onnx, f32_onnx)
    method = getattr(CalibrationMethod, "MIN_MAX_ASYM", None) or list(CalibrationMethod)[0]
    calibrator = Calibrator(f32, method)

    if not image_candidates:
        raise FileNotFoundError(
            "Calibration image not found.\n"
            "Use --calib-dir <dir> or --calib-image <path> with real image files.\n"
            f"Default fallback path also does not exist: {calib_image}\n"
            "Synthetic random calibration was removed because Furiosa quantizer may panic on non-representative inputs."
        )

    print(f"(calibration=images x{args.calib_iters}, source_count={len(image_candidates)})")
    for image_path in itertools.islice(itertools.cycle(image_candidates), args.calib_iters):
        sample = _load_calibration_sample(image_path, args.input_shape, np)
        print(
            f"(sample={image_path.name}, shape={sample.shape}, dtype={sample.dtype}, "
            f"min={sample.min():.6f}, max={sample.max():.6f}, mean={sample.mean():.6f})"
        )
        if float(np.max(np.abs(sample))) == 0.0:
            raise ValueError(
                f"Calibration sample is all zeros after preprocessing: {image_path}. "
                "Use a normal RGB photo for calibration."
            )
        calibrator.collect_data([[sample]])

    ranges = calibrator.compute_range()
    quantized = quantize(f32, ranges)
    quant_onnx.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(quantized, (bytes, bytearray)):
        quant_onnx.write_bytes(quantized)
    else:
        onnx.save(quantized, str(quant_onnx))
    print("onnx(quant) =", quant_onnx)

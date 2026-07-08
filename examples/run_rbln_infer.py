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


# ====== 경로 설정 (checkout root 기준, 컨테이너에서는 현재 마운트된 repo root) ======
ENGINE_PATH = REPO_ROOT / "builds" / "resnet50.rbln"   # <- builds 기준
IMG_PATH = REPO_ROOT / "tests" / "input.jpg"
LABELS_PATH = REPO_ROOT / "tests" / "imagenet_classes.txt"  # 있으면 사용, 없으면 cls_id만 출력


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
    parser = argparse.ArgumentParser(description="Run inference with a compiled RBLN model.")
    parser.add_argument("--engine-path", type=Path, default=ENGINE_PATH)
    parser.add_argument("--image", type=Path, default=IMG_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--output-name", default="output")
    parser.add_argument("--input-shape", type=_parse_shape, default=(1, 3, 224, 224))
    parser.add_argument("--device", type=int, default=int(os.getenv("RBLN_DEVICE", "0")))
    parser.add_argument("--tensor-type", choices=("np", "pt"), default="np")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--allow-dynamic-shape", action="store_true")
    return parser


def _check_files(engine_path: Path, image_path: Path):
    missing = []
    if not engine_path.is_file():
        missing.append(f"- engine: {engine_path}")
    if not image_path.is_file():
        missing.append(f"- image : {image_path}")
    if missing:
        raise FileNotFoundError("필요한 파일이 없습니다:\n" + "\n".join(missing))


# ====== ImageNet 표준 preprocess (weights 없이 고정) ======
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _load_labels(labels_path: Path):
    if labels_path.is_file():
        labels = [l.strip() for l in labels_path.read_text().splitlines() if l.strip()]
        return labels
    return None


if __name__ == "__main__":
    args = _build_parser().parse_args()

    try:
        import numpy as np
        import torch
        from torchvision.io.image import read_image
        from torchvision import transforms
    except ImportError:
        print("Error: 'numpy', 'torch', and 'torchvision' are required for the RBLN inference example.")
        sys.exit(1)

    from unified_sdk.types import RuntimeConfig
    from unified_sdk.runtime import create_runtime, infer, destroy_runtime

    engine_path = args.engine_path.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()

    if args.iters <= 0:
        raise ValueError("--iters must be > 0")

    _check_files(engine_path, image_path)
    labels = _load_labels(labels_path)

    preprocess = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    img = read_image(str(image_path))                 # [C,H,W], uint8
    batch_t = preprocess(img).unsqueeze(0)
    batch = batch_t if args.tensor_type == "pt" else batch_t.numpy()

    cfg = RuntimeConfig(
        backend="rbln",
        engine_path=str(engine_path),
        input_name=args.input_name,
        output_name=args.output_name,
        input_shape=args.input_shape,
        extra={
            "tensor_type": args.tensor_type,
            "device": args.device,
            "allow_dynamic_shape": args.allow_dynamic_shape,
        },
    )

    rh = create_runtime(cfg)

    x = batch
    _ = infer(rh, x)  # warmup

    times = []
    y = None
    for _ in range(args.iters):
        t0 = timeit.default_timer()
        y = infer(rh, x)
        t1 = timeit.default_timer()
        times.append((t1 - t0) * 1000)

    # y: numpy (1,1000) 가정
    y_t = torch.from_numpy(y)
    cls_id = int(torch.argmax(y_t, dim=1).item())

    if labels and 0 <= cls_id < len(labels):
        print(f"pred: {labels[cls_id]} (id={cls_id})")
    else:
        print(f"pred_id: {cls_id} (labels file not found: {labels_path})")

    print(f"Avg latency: {np.mean(times):.3f} ms, shape={y.shape}")
    print(f"(engine={engine_path}, tensor_type={args.tensor_type}, device={args.device})")

    destroy_runtime(rh)

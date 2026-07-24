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
ENGINE_PATH = REPO_ROOT / "builds" / "resnet50.mxq"   # <- builds 기준
IMG_PATH = REPO_ROOT / "models" / "input.jpg"
LABELS_PATH = REPO_ROOT / "models" / "labels.txt"  # 있으면 사용, 없으면 cls_id만 출력


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
    parser = argparse.ArgumentParser(description="Run inference with a compiled Mobilint ARISE (QB) .mxq model.")
    parser.add_argument("--engine-path", type=Path, default=ENGINE_PATH)
    parser.add_argument("--image", type=Path, default=IMG_PATH)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--output-name", default="output")
    parser.add_argument("--input-shape", type=_parse_shape, default=(224, 224, 3)) # 이 .mxq 는 정규화/레이아웃 변환을 내부에 포함해 컴파일되어 원본 uint8 HWC 입력을 기대함
    parser.add_argument("--device", type=int, default=int(os.getenv("MBLT_DEVICE", "0")))
    # 추론 단계는 로컬/직접 컴파일된 .mxq 의 실제 지원 코어 모드를 모르고 시작하는 경우가 많다.
    # 따라서 기본값은 가장 보수적인 `auto`로 두고, 필요할 때만 명시적으로 global4/global8 등을 준다.
    parser.add_argument("--core-mode", default=os.getenv("MBLT_CORE_MODE", "auto"))
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--allow-dynamic-shape", action="store_true")
    return parser


def _check_files(engine_path: Path):
    missing = []
    if not engine_path.is_file():
        missing.append(f"- engine: {engine_path}")
    if missing:
        raise FileNotFoundError("필요한 파일이 없습니다:\n" + "\n".join(missing))


# ====== ImageNet 표준 preprocess (weights 없이 고정) ======
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _dtype_to_numpy(data_type, np):
    name = str(data_type).lower()
    if "uint8" in name:
        return np.uint8
    if "int8" in name:
        return np.int8
    if "float16" in name:
        return np.float16
    return np.float32


def _prepare_image_batch_for_dtype(img_hwc_uint8, np, np_dtype):
    batch = np.ascontiguousarray(img_hwc_uint8)
    if np_dtype in (np.uint8, np.int8):
        return batch.astype(np_dtype, copy=False)

    # float 입력 모델은 smoke 기준으로 torchvision ImageNet 표준 정규화를 적용한다.
    batch = batch.astype(np_dtype) / np_dtype(255.0)
    mean = np.asarray(IMAGENET_MEAN, dtype=np_dtype)
    std = np.asarray(IMAGENET_STD, dtype=np_dtype)
    return np.ascontiguousarray((batch - mean) / std)


def _load_labels(labels_path: Path):
    if labels_path.is_file():
        labels = [l.strip() for l in labels_path.read_text().splitlines() if l.strip()]
        return labels
    return None


def to_mxq_input(arr, layout: str):
    """입력 배열을 이 .mxq 가 요구하는 HWC (H,W,C) uint8 로 정규화한다.

    레이아웃은 추측하지 않고 호출자가 명시한다 (데이터를 만든 쪽이 자기 레이아웃을 안다).
        layout: 'chw' | 'hwc' | 'nchw' | 'nhwc'
    torchvision.read_image 는 CHW 이므로 'chw', cv2/PIL(numpy) 는 보통 'hwc'.
    이 mxq 는 정규화가 내부에 포함되어 원본 uint8 을 기대하므로 dtype 은 uint8 로 맞춘다.
    """
    import numpy as np

    a = np.asarray(arr)
    key = layout.lower()
    if key == "nchw":
        a, key = a[0], "chw"
    elif key == "nhwc":
        a, key = a[0], "hwc"

    if key == "chw":
        a = np.transpose(a, (1, 2, 0))   # CHW -> HWC
    elif key != "hwc":
        raise ValueError(f"unknown layout: {layout!r} (use chw/hwc/nchw/nhwc)")

    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    return np.ascontiguousarray(a)


if __name__ == "__main__":
    args = _build_parser().parse_args()

    try:
        import numpy as np
        import torch
        from torchvision.io.image import read_image
        from torchvision import transforms
    except ImportError:
        print("Error: 'numpy', 'torch', and 'torchvision' are required for the QB inference example.")
        sys.exit(1)

    from unified_sdk.types import RuntimeConfig
    from unified_sdk.runtime import create_runtime, infer, destroy_runtime

    engine_path = args.engine_path.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()

    if args.iters <= 0:
        raise ValueError("--iters must be > 0")

    _check_files(engine_path)
    labels = _load_labels(labels_path)

    cfg = RuntimeConfig(
        backend="qb",
        engine_path=str(engine_path),
        input_name=args.input_name,
        output_name=args.output_name,
        input_shape=args.input_shape,
        extra={
            "device": args.device,
            "core_mode": args.core_mode,
            "allow_dynamic_shape": args.allow_dynamic_shape,
        },
    )

    rh = create_runtime(cfg)

    model = rh.ctx.get("model")
    input_dtype = getattr(model, "get_model_input_data_type", lambda: "Float32")()
    np_dtype = _dtype_to_numpy(input_dtype, np)

    if image_path.is_file():
        preprocess = transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
        ])
        img = read_image(str(image_path))                   # [C,H,W], uint8
        img = preprocess(img)                               # [3,224,224], uint8 유지
        batch_hwc = to_mxq_input(img.numpy(), layout="chw") # read_image 는 CHW -> HWC 로 변환
        batch = _prepare_image_batch_for_dtype(batch_hwc, np, np_dtype)
        input_source = f"{image_path} ({np_dtype.__name__})"
    else:
        # 합성 입력은 args.input_shape=(H,W,C) 기준으로 만들고, dtype 만 MXQ 메타에 맞춘다.
        batch = np.zeros(args.input_shape, dtype=np_dtype)
        input_source = f"synthetic zeros {args.input_shape} ({np_dtype.__name__})"

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
    y_t = torch.from_numpy(np.ascontiguousarray(y))
    cls_id = int(torch.argmax(y_t.reshape(y_t.shape[0], -1), dim=1).item())

    if labels and 0 <= cls_id < len(labels):
        print(f"pred: {labels[cls_id]} (id={cls_id})")
    else:
        print(f"pred_id: {cls_id} (labels file not found: {labels_path})")

    print(f"Avg latency: {np.mean(times):.3f} ms, shape={y.shape}")
    print(f"(engine={engine_path}, input={input_source}, core_mode={args.core_mode}, device={args.device}, input_dtype={input_dtype})")

    destroy_runtime(rh)

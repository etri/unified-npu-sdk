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

    # NOTE: 이 .mxq 는 qubee 컴파일 시 preprocess_dict(정규화 + CHW→HWC)를 모델에 포함한다.
    # 따라서 runtime 에는 정규화하지 않은 원본 uint8 을 넘기고, 레이아웃 변환은
    # to_mxq_input(arr, layout=...) 에 명시적으로 위임한다 (shape 추측 금지).
    # (get_model_input_shape()=[(224,224,3)], get_model_input_data_type()=Uint8)
    if image_path.is_file():
        preprocess = transforms.Compose([
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
        ])
        img = read_image(str(image_path))                 # [C,H,W], uint8
        img = preprocess(img)                             # [3,224,224], uint8 유지
        batch = to_mxq_input(img.numpy(), layout="chw")   # read_image 는 CHW -> HWC 로 변환
        input_source = str(image_path)
    else:
        # 합성 입력은 이미 args.input_shape=(H,W,C) 이므로 layout='hwc' (변환 없음)
        batch = to_mxq_input(torch.zeros(args.input_shape, dtype=torch.uint8).numpy(), layout="hwc")
        input_source = f"synthetic zeros {args.input_shape}"

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
    print(f"(engine={engine_path}, input={input_source}, core_mode={args.core_mode}, device={args.device})")

    destroy_runtime(rh)

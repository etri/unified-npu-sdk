import timeit
from pathlib import Path

import numpy as np
import torch
from torchvision.io.image import read_image
from torchvision import transforms

from unified_sdk.types import RuntimeConfig
from unified_sdk.runtime import create_runtime, infer, destroy_runtime


# ====== 경로 설정 (요청한 기준) ======
REPO_ROOT = Path("/workspace/unified-sdk")
ENGINE_PATH = REPO_ROOT / "builds" / "resnet50.rbln"   # <- builds 기준
IMG_PATH = REPO_ROOT / "tests" / "input.jpg"
LABELS_PATH = REPO_ROOT / "tests" / "imagenet_classes.txt"  # 있으면 사용, 없으면 cls_id만 출력


def _check_files():
    missing = []
    if not ENGINE_PATH.is_file():
        missing.append(f"- engine: {ENGINE_PATH}")
    if not IMG_PATH.is_file():
        missing.append(f"- image : {IMG_PATH}")
    if missing:
        raise FileNotFoundError("필요한 파일이 없습니다:\n" + "\n".join(missing))


# ====== ImageNet 표준 preprocess (weights 없이 고정) ======
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

preprocess = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ConvertImageDtype(torch.float32),   # uint8 -> float32, [0,1]
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _load_labels():
    if LABELS_PATH.is_file():
        labels = [l.strip() for l in LABELS_PATH.read_text().splitlines() if l.strip()]
        return labels
    return None


if __name__ == "__main__":
    _check_files()
    labels = _load_labels()

    img = read_image(str(IMG_PATH))                 # [C,H,W], uint8
    batch = preprocess(img).unsqueeze(0).numpy()    # ✅ weights.transforms() 대신 preprocess 사용

    cfg = RuntimeConfig(
        backend="rbln",
        engine_path=str(ENGINE_PATH),   # ✅ builds/resnet50.rbln
        input_name="input",
        output_name="output",
        input_shape=(1, 3, 224, 224),
        extra={"tensor_type": "np"},    # numpy 입력
    )

    rh = create_runtime(cfg)

    x = batch
    _ = infer(rh, x)  # warmup

    iters = 50
    times = []
    y = None
    for _ in range(iters):
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
        print(f"pred_id: {cls_id} (labels file not found: {LABELS_PATH})")

    print(f"Avg latency: {np.mean(times):.3f} ms, shape={y.shape}")

    destroy_runtime(rh)



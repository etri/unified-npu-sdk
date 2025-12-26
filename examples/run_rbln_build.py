# examples/run_rbln_build.py
from pathlib import Path
import os

import torch
from torchvision.models import resnet50

from unified_sdk.types import BuildConfig
from unified_sdk.build.api import build_unified


def _resolve_repo_root() -> Path:
    """
    기준:
      1) /workspace/unified-sdk 가 있으면 그걸 사용
      2) 없으면 현재 파일 위치(.../examples/run_rbln_build.py) 기준으로 repo root 추론
    """
    ws_root = Path("/workspace/unified-sdk")
    if ws_root.is_dir():
        return ws_root

    # examples/ 아래에 이 파일이 있다는 가정
    return Path(__file__).resolve().parents[1]


REPO_ROOT = _resolve_repo_root()
EXAMPLES_DIR = REPO_ROOT / "examples"
MODELS_DIR = REPO_ROOT / "models"
BUILDS_DIR = REPO_ROOT / "builds"


def _check_repo_layout():
    missing = []
    if not EXAMPLES_DIR.is_dir():
        missing.append(str(EXAMPLES_DIR))
    if not MODELS_DIR.is_dir():
        missing.append(str(MODELS_DIR))

    # builds는 없으면 만들어줌 (완화)
    BUILDS_DIR.mkdir(parents=True, exist_ok=True)

    if missing:
        raise FileNotFoundError(
            "필수 폴더가 없습니다:\n"
            + "\n".join(f"- {p}" for p in missing)
            + f"\n\n(현재 기준 REPO_ROOT = {REPO_ROOT})"
        )


def _find_weights(models_dir: Path) -> Path:
    candidates = sorted(models_dir.glob("resnet50*.pth")) + sorted(models_dir.glob("resnet50*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"{models_dir} 에서 resnet50 가중치 파일을 찾지 못했습니다.\n"
            f"예) {models_dir/'resnet50.pth'} 또는 {models_dir/'resnet50_state_dict.pth'}"
        )
    return candidates[0]


def _load_state_dict(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu")
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
    _check_repo_layout()

    weights_path = _find_weights(MODELS_DIR)

    # 다운로드 트리거 제거 (로컬 state_dict만 사용)
    model = resnet50(weights=None)
    sd = _load_state_dict(weights_path)
    model.load_state_dict(sd, strict=False)
    model.eval()

    cfg = BuildConfig(
        backend="rbln",
        model_or_path=model,
        out_dir=str(BUILDS_DIR),   # /workspace/unified-sdk/builds
        model_name="resnet50",
        precision="fp32",
        input_name="input",
        min_input_shape=(1, 3, 224, 224),
        opt_input_shape=(1, 3, 224, 224),
        max_input_shape=(1, 3, 224, 224),
    )

    result = build_unified(cfg)
    print("Complete!", result.compiled_model_path)
    print(f"(repo_root={REPO_ROOT})")


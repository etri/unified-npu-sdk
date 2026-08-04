from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any


def find_local_mxq(models_dir: Path, model_name: str) -> Path | None:
    candidates = sorted(models_dir.glob(f"{model_name}*.mxq")) + sorted(models_dir.glob("*.mxq"))
    return candidates[0] if candidates else None


def normalize_mxq_into_models(src_mxq: Path, models_dir: Path, model_name: str) -> Path:
    target_name = model_name if model_name.lower().endswith(".mxq") else f"{model_name}.mxq"
    target = models_dir / target_name
    models_dir.mkdir(parents=True, exist_ok=True)
    if src_mxq.resolve() != target.resolve():
        shutil.copyfile(src_mxq, target)
    return target


def _normalize_model_zoo_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _resolve_model_zoo_class(model_name: str):
    try:
        from mblt_model_zoo import vision
    except Exception:
        return None

    normalized = _normalize_model_zoo_name(model_name)
    matches: list[tuple[str, Any]] = []
    for attr_name in dir(vision):
        if attr_name.startswith("_"):
            continue
        candidate = getattr(vision, attr_name, None)
        if candidate is None or not callable(candidate):
            continue
        if _normalize_model_zoo_name(attr_name) == normalized:
            matches.append((attr_name, candidate))

    if not matches:
        return None

    matches.sort(key=lambda item: (not item[0][:1].isupper(), item[0]))
    return matches[0]


def list_model_zoo_models() -> list[tuple[str, str]]:
    try:
        from mblt_model_zoo import vision
    except Exception:
        return []

    seen: set[str] = set()
    items: list[tuple[str, str]] = []
    for attr_name in dir(vision):
        if attr_name.startswith("_"):
            continue
        candidate = getattr(vision, attr_name, None)
        if candidate is None or not callable(candidate):
            continue
        normalized = _normalize_model_zoo_name(attr_name)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        items.append((normalized, attr_name))
    items.sort()
    return items


def find_model_zoo_mxq(model_name: str, product: str, core_mode: str) -> Path | None:
    home = Path.home()
    normalized = model_name.lower()
    zoo_root_new = home / ".mblt_model_zoo" / "vision" / product / core_mode
    zoo_root_legacy = home / ".mblt_model_zoo" / product

    explicit_candidates = [
        zoo_root_new / f"{normalized}_IMAGENET1K_V2.mxq",
        zoo_root_new / f"{normalized}_DEFAULT.mxq",
        zoo_root_legacy / f"{normalized}_IMAGENET1K_V2.mxq",
        zoo_root_legacy / f"{normalized}_DEFAULT.mxq",
    ]
    for candidate in explicit_candidates:
        if candidate.is_file():
            return candidate

    glob_candidates = (
        sorted(zoo_root_new.glob(f"{normalized}*.mxq"))
        + sorted(zoo_root_legacy.glob(f"{normalized}*.mxq"))
    )
    if glob_candidates:
        return glob_candidates[0]

    recursive_candidates = sorted((home / ".mblt_model_zoo").rglob(f"{normalized}*.mxq"))
    return recursive_candidates[0] if recursive_candidates else None


def trigger_model_zoo_fetch(model_name: str, product: str, core_mode: str, models_dir: Path) -> Path | None:
    """Best-effort materialization of a standard MXQ via mblt_model_zoo."""
    try:
        import torch
        from torchvision.io import write_png
    except Exception:
        return None

    resolved = _resolve_model_zoo_class(model_name)
    if resolved is None:
        return None
    _, model_cls = resolved

    scratch_image = models_dir / f"_mblt_model_zoo_{model_name.lower()}_smoke.png"
    if not scratch_image.is_file():
        scratch_image.parent.mkdir(parents=True, exist_ok=True)
        img = torch.full((3, 224, 224), 127, dtype=torch.uint8)
        write_png(img, str(scratch_image))

    ctor_variants = (
        {
            "local_path": None,
            "model_type": "DEFAULT",
            "infer_mode": core_mode,
            "product": product,
        },
        {
            "model_type": "DEFAULT",
            "infer_mode": core_mode,
            "product": product,
        },
        {
            "infer_mode": core_mode,
            "product": product,
        },
        {},
    )
    model = None
    for kwargs in ctor_variants:
        try:
            model = model_cls(**kwargs)
            break
        except TypeError:
            continue
    if model is None:
        return None

    try:
        if hasattr(model, "preprocess"):
            input_img = model.preprocess(str(scratch_image))
            output = model(input_img)
            try:
                model.postprocess(output)
            except Exception:
                pass
        elif callable(model):
            try:
                model(str(scratch_image))
            except Exception:
                try:
                    model()
                except Exception:
                    return None
        else:
            return None
    finally:
        try:
            model.dispose()
        except Exception:
            pass

    return find_model_zoo_mxq(model_name, product, core_mode)

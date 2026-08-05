from __future__ import annotations

from pathlib import Path
import re


def _normalize_model_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def find_local_enf(models_dir: Path, model_name: str) -> Path | None:
    normalized = _normalize_model_name(model_name)
    for candidate in sorted(models_dir.glob("*.enf")):
        if _normalize_model_name(candidate.stem) == normalized:
            return candidate
    return None


def list_model_zoo_targets() -> list[str]:
    try:
        from furiosa.models import vision
    except Exception:
        return []

    declared = getattr(vision, "__all__", None)
    if isinstance(declared, (list, tuple)):
        return sorted({str(name) for name in declared if isinstance(name, str) and not name.startswith("_")})

    return sorted({name for name in dir(vision) if not name.startswith("_")})


def resolve_model_zoo_target(model_name: str) -> str | None:
    normalized = _normalize_model_name(model_name)
    for candidate in list_model_zoo_targets():
        if _normalize_model_name(candidate) == normalized:
            return candidate
    return None


def fetch_model_zoo_enf(model_name: str, target_npu: str, models_dir: Path) -> Path | None:
    try:
        from furiosa.models import vision
    except Exception:
        return None

    resolved = resolve_model_zoo_target(model_name)
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

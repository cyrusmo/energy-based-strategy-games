"""Device selection helpers for small PyTorch experiment jobs."""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

DEFAULT_CALIBRATION_PATH = Path("outputs/public/device_calibration.json")
_WARNED_FALLBACKS: set[str] = set()


def resolve_device(preference: str | torch.device | None = "auto", job: str | None = None) -> torch.device:
    """Resolve a user preference to a concrete torch device.

    ``auto`` first honors a calibration artifact for the named job and then
    falls back to MPS when it is available. Explicit ``mps`` requests degrade to
    CPU with a single warning instead of failing on non-Apple or CPU-only builds.
    """

    if isinstance(preference, torch.device):
        preference = preference.type
    requested = str(preference or "auto").lower()
    if requested == "auto":
        calibrated = recommended_device_for_job(job) if job else None
        if calibrated:
            return resolve_device(calibrated)
        return torch.device("mps" if mps_is_available() else "cpu")
    if requested == "mps":
        if mps_is_available():
            return torch.device("mps")
        _warn_once("mps-unavailable", "MPS requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        _warn_once("cuda-unavailable", "CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device preference: {preference!r}")


def mps_is_available() -> bool:
    """Return true when this PyTorch build can execute MPS kernels."""

    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built() and torch.backends.mps.is_available())


def move_models(*modules: nn.Module, device: str | torch.device) -> tuple[nn.Module, ...]:
    """Move modules to ``device`` and keep parameters in float32."""

    resolved = resolve_device(device)
    for module in modules:
        module.to(device=resolved, dtype=torch.float32)
    return modules


def tensor_device(tensor: torch.Tensor | None, fallback: str | torch.device = "cpu") -> torch.device:
    """Return a tensor's device, or a resolved fallback for missing tensors."""

    return tensor.device if tensor is not None else resolve_device(fallback)


def recommended_device_map(path: str | Path = DEFAULT_CALIBRATION_PATH) -> dict[str, str]:
    """Read a calibration artifact and return ``job -> recommended_device``."""

    calibration_path = Path(path)
    if not calibration_path.exists():
        return {}
    try:
        with calibration_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    jobs = payload.get("jobs", payload)
    if isinstance(jobs, Mapping):
        iterable: Iterable[Any] = jobs.values()
    elif isinstance(jobs, list):
        iterable = jobs
    else:
        return {}
    recommendations: dict[str, str] = {}
    for item in iterable:
        if not isinstance(item, Mapping):
            continue
        job = item.get("job")
        device = item.get("recommended_device")
        if isinstance(job, str) and isinstance(device, str):
            recommendations[job] = device
    return recommendations


def recommended_device_for_job(job: str | None, path: str | Path = DEFAULT_CALIBRATION_PATH) -> str | None:
    """Return the calibrated recommendation for a single job, if present."""

    if not job:
        return None
    return recommended_device_map(path).get(job)


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED_FALLBACKS:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        _WARNED_FALLBACKS.add(key)

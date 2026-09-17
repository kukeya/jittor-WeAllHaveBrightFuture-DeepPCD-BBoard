"""Portable, data-free model configuration for checkpoint inference."""

from __future__ import annotations

import json
from pathlib import Path


INFERENCE_CONFIG_FORMAT = "pcdenoise_inference_config_v1"


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def load_inference_config(path: Path) -> dict[str, object]:
    """Load the model-only subset needed to reconstruct one checkpoint."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid inference config: {path}") from error
    if not isinstance(document, dict):
        raise ValueError("inference config root must be an object")
    if document.get("format") != INFERENCE_CONFIG_FORMAT:
        raise ValueError(
            f"inference config format must be {INFERENCE_CONFIG_FORMAT}"
        )
    model = document.get("model")
    data = document.get("data")
    use_cuda = document.get("use_cuda")
    if not isinstance(model, dict) or not model:
        raise ValueError("inference config model must be a nonempty object")
    if not isinstance(data, dict):
        raise ValueError("inference config data must be an object")
    if not isinstance(use_cuda, bool):
        raise ValueError("inference config use_cuda must be a bool")
    checkpoint_step = document.get("checkpoint_step")
    if (
        isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step < 0
    ):
        raise ValueError("inference config checkpoint_step must be nonnegative")
    return {
        "format": INFERENCE_CONFIG_FORMAT,
        "canonical_config_sha256": _sha256(
            document.get("canonical_config_sha256"),
            name="canonical_config_sha256",
        ),
        "checkpoint_sha256": _sha256(
            document.get("checkpoint_sha256"),
            name="checkpoint_sha256",
        ),
        "checkpoint_step": checkpoint_step,
        "model": model,
        "data": data,
        "use_cuda": use_cuda,
    }


__all__ = ["INFERENCE_CONFIG_FORMAT", "load_inference_config"]

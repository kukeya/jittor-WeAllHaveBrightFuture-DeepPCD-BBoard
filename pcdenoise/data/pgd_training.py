"""Source-faithful supervised patch generation for the PGD baseline."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .mesh_dataset import fit_unit_sphere
from .noise import (
    DEFAULT_SIGMA_RANGE,
    add_noise,
    rotate_clean_noisy,
)

TRAINING_NORMALIZATION_MODES = ("clean_unit", "noisy_max")


def _cloud(values: object, *, name: str) -> np.ndarray:
    cloud = np.asarray(values)
    if cloud.ndim != 2 or cloud.shape[0] == 0 or cloud.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), N > 0")
    if not np.issubdtype(cloud.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    result = np.ascontiguousarray(cloud, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _patch_size(value: object, *, point_count: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError("patch_size must be a positive integer")
    count = int(value)
    if count > point_count:
        raise ValueError("patch_size must not exceed the cloud size")
    return count


@dataclass(frozen=True)
class PGDTrainingCloud:
    """The exact whole-cloud preprocessing prefix shared by training tools."""

    whole_noisy: np.ndarray
    whole_clean: np.ndarray
    noise_profile: str
    noise_scale: float
    source_noise_scale: float
    rotation: np.ndarray
    rotation_mode: str = "euler_xyz"
    normalization_mode: str = "clean_unit"
    normalization_center: np.ndarray | None = None
    normalization_scale: float = 1.0


@dataclass(frozen=True)
class PGDTrainingPatch:
    """One centered pair plus its reproducibility metadata."""

    noisy: np.ndarray
    clean: np.ndarray
    indices: np.ndarray
    center_index: int
    seed_point: np.ndarray
    noise_profile: str
    noise_scale: float
    source_noise_scale: float
    rotation: np.ndarray
    whole_noisy: np.ndarray
    whole_clean: np.ndarray
    rotation_mode: str = "euler_xyz"
    normalization_mode: str = "clean_unit"
    normalization_center: np.ndarray | None = None
    normalization_scale: float = 1.0
    patch_scale_applied: bool = False

    @property
    def sigma(self) -> float:
        """Legacy read-only alias for Gaussian-era callers."""

        return self.noise_scale


def prepare_code_faithful_training_cloud(
    clean_unit_sphere: np.ndarray,
    rng: np.random.Generator,
    *,
    noise_profile: str = "gaussian",
    noise_scale: float | None = None,
    noise_scale_range: tuple[float, float] | None = None,
    sigma: float | None = None,
    sigma_range: tuple[float, float] | None = None,
    rotate: bool = True,
    rotation_mode: str = "euler_xyz",
    normalization_mode: str = "clean_unit",
) -> PGDTrainingCloud:
    """Run the training preprocessing prefix before center selection/KNN.

    Every random draw and float conversion here is shared with patch
    generation.  Diagnostics may therefore inspect the exact effective
    noise-scale teacher without paying for center selection or noisy-space
    KNN, neither of which can alter that teacher.
    """

    clean = _cloud(clean_unit_sphere, name="clean_unit_sphere")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    if normalization_mode not in TRAINING_NORMALIZATION_MODES:
        raise ValueError(
            "normalization_mode must be one of "
            f"{TRAINING_NORMALIZATION_MODES}, got {normalization_mode!r}"
        )
    if noise_scale is not None and sigma is not None:
        raise ValueError("noise_scale and sigma cannot both be supplied")
    if noise_scale_range is not None and sigma_range is not None:
        raise ValueError(
            "noise_scale_range and sigma_range cannot both be supplied"
        )
    actual_scale_argument = (
        noise_scale if noise_scale is not None else sigma
    )
    actual_scale_range = (
        noise_scale_range
        if noise_scale_range is not None
        else sigma_range
        if sigma_range is not None
        else DEFAULT_SIGMA_RANGE
    )
    noisy, actual_noise_scale = add_noise(
        clean,
        rng,
        profile=noise_profile,
        scale=actual_scale_argument,
        scale_range=actual_scale_range,
    )
    whole_clean, whole_noisy, rotation = rotate_clean_noisy(
        clean,
        noisy,
        rng,
        enabled=rotate,
        mode=rotation_mode,
    )
    if normalization_mode == "noisy_max":
        transform = fit_unit_sphere(whole_noisy)
        whole_clean = transform.apply(whole_clean)
        whole_noisy = transform.apply(whole_noisy)
        normalization_center = transform.center
        normalization_scale = transform.scale
    else:
        normalization_center = np.zeros(3, dtype=np.float64)
        normalization_scale = 1.0
    effective_noise_scale = float(
        actual_noise_scale / normalization_scale
    )
    if (
        not np.isfinite(effective_noise_scale)
        or effective_noise_scale <= 0.0
    ):
        raise RuntimeError(
            "effective noise scale after normalization is invalid"
        )
    return PGDTrainingCloud(
        whole_noisy=np.ascontiguousarray(whole_noisy),
        whole_clean=np.ascontiguousarray(whole_clean),
        noise_profile=noise_profile,
        noise_scale=effective_noise_scale,
        source_noise_scale=float(actual_noise_scale),
        rotation=np.ascontiguousarray(rotation, dtype=np.float32),
        rotation_mode=rotation_mode,
        normalization_mode=normalization_mode,
        normalization_center=np.ascontiguousarray(
            normalization_center,
            dtype=np.float64,
        ),
        normalization_scale=float(normalization_scale),
    )


def sample_code_faithful_training_patch(
    clean_unit_sphere: np.ndarray,
    rng: np.random.Generator,
    *,
    patch_size: int = 1000,
    noise_profile: str = "gaussian",
    noise_scale: float | None = None,
    noise_scale_range: tuple[float, float] | None = None,
    sigma: float | None = None,
    sigma_range: tuple[float, float] | None = None,
    rotate: bool = True,
    rotation_mode: str = "euler_xyz",
    normalization_mode: str = "clean_unit",
    center_index: int | None = None,
) -> PGDTrainingPatch:
    """Add whole-cloud noise/rotation, then gather a noisy-space KNN patch.

    This intentionally performs only seed centering.  The fixed PGD source
    does *not* divide a training patch by its local radius after the whole
    clean cloud has already been normalized to the unit sphere.
    """

    clean = _cloud(clean_unit_sphere, name="clean_unit_sphere")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    if normalization_mode not in TRAINING_NORMALIZATION_MODES:
        raise ValueError(
            "normalization_mode must be one of "
            f"{TRAINING_NORMALIZATION_MODES}, got {normalization_mode!r}"
        )
    count = _patch_size(patch_size, point_count=len(clean))
    prepared = prepare_code_faithful_training_cloud(
        clean,
        rng,
        noise_profile=noise_profile,
        noise_scale=noise_scale,
        noise_scale_range=noise_scale_range,
        sigma=sigma,
        sigma_range=sigma_range,
        rotate=rotate,
        rotation_mode=rotation_mode,
        normalization_mode=normalization_mode,
    )
    whole_clean = prepared.whole_clean
    whole_noisy = prepared.whole_noisy

    if center_index is None:
        selected_center = int(rng.integers(0, len(clean)))
    else:
        if (
            isinstance(center_index, bool)
            or not isinstance(center_index, Integral)
        ):
            raise IndexError("center_index must be an integer")
        selected_center = int(center_index)
        if not 0 <= selected_center < len(clean):
            raise IndexError("center_index is out of range")

    seed_point = whole_noisy[selected_center].copy()
    delta = (
        whole_noisy.astype(np.float64)
        - seed_point.astype(np.float64)
    )
    squared = np.einsum("ij,ij->i", delta, delta)
    if count == len(clean):
        candidates = np.arange(len(clean), dtype=np.int64)
    else:
        candidates = np.argpartition(squared, count - 1)[:count]
    order = np.lexsort((candidates, squared[candidates]))
    indices = np.ascontiguousarray(
        candidates[order],
        dtype=np.int64,
    )

    noisy_patch = np.ascontiguousarray(
        whole_noisy[indices] - seed_point,
        dtype=np.float32,
    )
    clean_patch = np.ascontiguousarray(
        whole_clean[indices] - seed_point,
        dtype=np.float32,
    )
    if not (
        np.isfinite(noisy_patch).all()
        and np.isfinite(clean_patch).all()
    ):
        raise RuntimeError("PGD patch centering produced non-finite values")
    return PGDTrainingPatch(
        noisy=noisy_patch,
        clean=clean_patch,
        indices=indices,
        center_index=selected_center,
        seed_point=seed_point,
        noise_profile=noise_profile,
        noise_scale=prepared.noise_scale,
        source_noise_scale=prepared.source_noise_scale,
        rotation=prepared.rotation,
        whole_noisy=np.ascontiguousarray(whole_noisy),
        whole_clean=np.ascontiguousarray(whole_clean),
        rotation_mode=rotation_mode,
        normalization_mode=normalization_mode,
        normalization_center=np.ascontiguousarray(
            prepared.normalization_center,
            dtype=np.float64,
        ),
        normalization_scale=prepared.normalization_scale,
        patch_scale_applied=False,
    )


__all__ = [
    "PGDTrainingCloud",
    "PGDTrainingPatch",
    "TRAINING_NORMALIZATION_MODES",
    "prepare_code_faithful_training_cloud",
    "sample_code_faithful_training_patch",
]

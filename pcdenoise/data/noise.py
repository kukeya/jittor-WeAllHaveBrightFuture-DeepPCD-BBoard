"""Faithful first-baseline noise and paired rigid augmentation."""

from __future__ import annotations

import itertools
from typing import Tuple

import numpy as np


DEFAULT_SIGMA_RANGE = (0.005, 0.020)
NOISE_PROFILES = ("gaussian", "starter_laplace")
ROTATION_MODES = ("euler_xyz", "cube24")


def _points(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 2 or result.shape[1] != 3 or result.shape[0] == 0:
        raise ValueError(f"{name} must have shape (N, 3), N > 0")
    if not np.issubdtype(result.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    result = result.astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _generator(rng: np.random.Generator) -> np.random.Generator:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    return rng


def _nonnegative_level(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and nonnegative")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and nonnegative") from error
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _increasing_level_range(
    values: Tuple[float, float],
    name: str,
) -> tuple[float, float]:
    try:
        if len(values) != 2:
            raise ValueError
        raw_low, raw_high = values
        if isinstance(raw_low, (bool, np.bool_)) or isinstance(
            raw_high, (bool, np.bool_)
        ):
            raise ValueError
        low, high = float(raw_low), float(raw_high)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{name} must contain exactly two finite values"
        ) from error
    if not np.isfinite([low, high]).all() or low < 0.0 or high <= low:
        raise ValueError(
            f"{name} must be finite, nonnegative, and increasing"
        )
    return low, high


def add_isotropic_gaussian(
    clean: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma: float | None = None,
    sigma_range: Tuple[float, float] = DEFAULT_SIGMA_RANGE,
) -> tuple[np.ndarray, float]:
    """Add one isotropic Gaussian perturbation per corresponding point.

    When ``sigma`` is omitted, its actual value is sampled uniformly from
    ``sigma_range`` using the supplied generator and returned with the noisy
    cloud.  No secondary or mixture noise strategy is applied.
    """

    clean_points = _points(clean, "clean")
    generator = _generator(rng)
    low, high = _increasing_level_range(sigma_range, "sigma_range")
    if sigma is None:
        actual_sigma = float(generator.uniform(low, high))
    else:
        actual_sigma = _nonnegative_level(sigma, "sigma")

    perturbation = generator.normal(
        loc=0.0, scale=actual_sigma, size=clean_points.shape
    )
    noisy = clean_points.astype(np.float64) + perturbation
    result = noisy.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Gaussian perturbation produced non-finite points")
    return result, actual_sigma


def add_starter_laplace(
    clean: np.ndarray,
    rng: np.random.Generator,
    *,
    scale: float | None = None,
    scale_range: Tuple[float, float] = DEFAULT_SIGMA_RANGE,
) -> tuple[np.ndarray, float]:
    """Add noise with the official Starter's sampled Laplace scale.

    The sampled or fixed value is the Laplace distribution's scale ``b``;
    it is not converted to a standard deviation.
    """

    clean_points = _points(clean, "clean")
    generator = _generator(rng)
    low, high = _increasing_level_range(scale_range, "scale_range")
    if scale is None:
        actual_scale = float(generator.uniform(low, high))
    else:
        actual_scale = _nonnegative_level(scale, "scale")

    perturbation = generator.laplace(
        loc=0.0,
        scale=actual_scale,
        size=clean_points.shape,
    )
    noisy = clean_points.astype(np.float64) + perturbation
    result = noisy.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Laplace perturbation produced non-finite points")
    return result, actual_scale


def add_noise(
    clean: np.ndarray,
    rng: np.random.Generator,
    *,
    profile: str,
    scale: float | None = None,
    scale_range: Tuple[float, float] | None = None,
    level: float | None = None,
    level_range: Tuple[float, float] | None = None,
) -> tuple[np.ndarray, float]:
    """Apply a named profile and return noisy points plus its actual scale.

    ``scale`` and ``scale_range`` are the canonical generic keywords.
    ``level`` and ``level_range`` remain accepted aliases for callers that
    already use the initial profile API.
    """

    uses_scale_family = scale is not None or scale_range is not None
    uses_level_family = level is not None or level_range is not None
    if uses_scale_family and uses_level_family:
        raise ValueError(
            "cannot mix scale/scale_range with level/level_range"
        )
    actual_scale = scale if scale is not None else level
    actual_scale_range = (
        scale_range
        if scale_range is not None
        else level_range
        if level_range is not None
        else DEFAULT_SIGMA_RANGE
    )
    if profile == "gaussian":
        return add_isotropic_gaussian(
            clean,
            rng,
            sigma=actual_scale,
            sigma_range=actual_scale_range,
        )
    if profile == "starter_laplace":
        return add_starter_laplace(
            clean,
            rng,
            scale=actual_scale,
            scale_range=actual_scale_range,
        )
    raise ValueError(
        f"unknown noise profile {profile!r}; expected one of {NOISE_PROFILES}"
    )


def _cube24_rotation_matrices() -> tuple[np.ndarray, ...]:
    """Return the 24 proper signed-permutation rotations in stable order."""

    matrices = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            matrix = np.zeros((3, 3), dtype=np.float64)
            matrix[np.arange(3), permutation] = signs
            if np.linalg.det(matrix) > 0.0:
                matrices.append(matrix.astype(np.float32))
    matrices.sort(key=lambda matrix: tuple(matrix.reshape(-1).tolist()))
    if len(matrices) != 24:
        raise RuntimeError("cube24 rotation construction is invalid")
    return tuple(matrices)


_CUBE24_ROTATIONS = _cube24_rotation_matrices()


def rotate_clean_noisy(
    clean: np.ndarray,
    noisy: np.ndarray,
    rng: np.random.Generator,
    *,
    enabled: bool = True,
    mode: str = "euler_xyz",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one shared random rotation to a paired cloud.

    ``euler_xyz`` preserves the historical RNG and arithmetic path.
    ``cube24`` samples uniformly from the proper signed-permutation group,
    which preserves an axis-aligned iid Laplace noise law exactly.
    """

    clean_points = _points(clean, "clean")
    noisy_points = _points(noisy, "noisy")
    if clean_points.shape != noisy_points.shape:
        raise ValueError("clean and noisy must have identical shapes")
    generator = _generator(rng)
    if mode not in ROTATION_MODES:
        raise ValueError(
            f"rotation mode must be one of {ROTATION_MODES}, got {mode!r}"
        )
    if not enabled:
        return (
            clean_points.copy(),
            noisy_points.copy(),
            np.eye(3, dtype=np.float32),
        )

    if mode == "cube24":
        rotation = _CUBE24_ROTATIONS[
            int(generator.integers(0, len(_CUBE24_ROTATIONS)))
        ].astype(np.float64)
    else:
        angle_x, angle_y, angle_z = generator.uniform(
            0.0, 2.0 * np.pi, size=3
        )
        cosine_x, sine_x = np.cos(angle_x), np.sin(angle_x)
        cosine_y, sine_y = np.cos(angle_y), np.sin(angle_y)
        cosine_z, sine_z = np.cos(angle_z), np.sin(angle_z)
        rotation_x = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, cosine_x, -sine_x],
                [0.0, sine_x, cosine_x],
            ],
            dtype=np.float64,
        )
        rotation_y = np.asarray(
            [
                [cosine_y, 0.0, sine_y],
                [0.0, 1.0, 0.0],
                [-sine_y, 0.0, cosine_y],
            ],
            dtype=np.float64,
        )
        rotation_z = np.asarray(
            [
                [cosine_z, -sine_z, 0.0],
                [sine_z, cosine_z, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rotation = rotation_z @ rotation_y @ rotation_x
    with np.errstate(over="ignore", invalid="ignore"):
        clean_rotated = (
            clean_points.astype(np.float64) @ rotation.T
        ).astype(np.float32)
        noisy_rotated = (
            noisy_points.astype(np.float64) @ rotation.T
        ).astype(np.float32)
    if not (
        np.isfinite(clean_rotated).all()
        and np.isfinite(noisy_rotated).all()
    ):
        raise ValueError("rotation produced non-finite points")
    return clean_rotated, noisy_rotated, rotation.astype(np.float32)


__all__ = [
    "DEFAULT_SIGMA_RANGE",
    "NOISE_PROFILES",
    "ROTATION_MODES",
    "add_isotropic_gaussian",
    "add_starter_laplace",
    "add_noise",
    "rotate_clean_noisy",
]

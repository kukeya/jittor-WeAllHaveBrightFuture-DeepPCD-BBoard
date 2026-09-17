"""Competition-faithful CPU Chamfer distance.

The organizer evaluator uses SciPy cKDTree, squared Euclidean distance, and a
reference-derived unit-sphere transform.  This module deliberately mirrors
those semantics instead of using a differentiable training approximation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree


DEGENERATE_SCALE_THRESHOLD = 1.0e-12


def as_float64_points(values: np.ndarray, name: str) -> np.ndarray:
    """Validate a non-empty point cloud and return a float64 view/copy."""

    points = np.asarray(values)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        raise ValueError(f"{name} must have finite shape (N, 3), N > 0")
    if not np.issubdtype(points.dtype, np.number) or np.issubdtype(
        points.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must be real numeric")
    points = points.astype(np.float64, copy=False)
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite values")
    return points


def _reference_center_scale(reference: np.ndarray) -> Tuple[np.ndarray, float]:
    center = (reference.max(axis=0) + reference.min(axis=0)) / 2.0
    centered = reference - center
    scale = float(np.sqrt(np.square(centered).sum(axis=1)).max())
    return center, scale


def fit_reference_normalization(
    reference_points: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Fit bbox-center/max-radius parameters while preserving float64 precision."""

    reference = as_float64_points(reference_points, "reference_points")
    center, scale = _reference_center_scale(reference)
    if not np.isfinite(scale) or scale < DEGENERATE_SCALE_THRESHOLD:
        raise ValueError("reference point cloud has degenerate normalization scale")
    return center.copy(), scale


def apply_reference_normalization(
    points: np.ndarray,
    center: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Apply an already-fitted reference transform without refitting."""

    values = as_float64_points(points, "points")
    actual_center = np.asarray(center, dtype=np.float64)
    actual_scale = float(scale)
    if actual_center.shape != (3,) or not np.isfinite(actual_center).all():
        raise ValueError("center must be a finite length-three vector")
    if (
        not np.isfinite(actual_scale)
        or actual_scale < DEGENERATE_SCALE_THRESHOLD
    ):
        raise ValueError("scale must be finite and non-degenerate")
    result = (values - actual_center) / actual_scale
    if not np.isfinite(result).all():
        raise ValueError("normalization produced non-finite values")
    return result


def chamfer_distance(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    normalize: bool = True,
) -> float:
    """Return bidirectional mean squared nearest-neighbor distance."""

    predicted = as_float64_points(prediction, "prediction")
    target = as_float64_points(reference, "reference")

    if normalize:
        center, scale = _reference_center_scale(target)
        # This edge behavior is copied from the organizer evaluator.
        if scale < DEGENERATE_SCALE_THRESHOLD:
            return 0.0
        target = (target - center) / scale
        predicted = (predicted - center) / scale

    target_tree = cKDTree(target)
    predicted_to_target, _ = target_tree.query(predicted, k=1)
    prediction_tree = cKDTree(predicted)
    target_to_predicted, _ = prediction_tree.query(target, k=1)
    value = np.square(predicted_to_target).mean() + np.square(
        target_to_predicted
    ).mean()
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise RuntimeError(f"Chamfer distance is invalid: {result}")
    return result

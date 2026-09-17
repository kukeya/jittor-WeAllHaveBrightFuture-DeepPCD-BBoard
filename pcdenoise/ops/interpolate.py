"""Inverse-Euclidean-distance feature interpolation in pure Jittor."""

from __future__ import annotations

import math
from numbers import Real

import jittor as jt

from .indexing import _index_points_unchecked, _validate_batched_points
from .knn import knn


def inverse_distance_interpolate(
    query: jt.Var,
    reference: jt.Var,
    reference_features: jt.Var,
    *,
    k: int = 8,
    query_block_size: int = 256,
    reference_block_size: int = 1024,
    epsilon: float = 1.0e-12,
    return_details: bool = False,
    validate_finite: bool = True,
) -> jt.Var | tuple[jt.Var, jt.Var, jt.Var, jt.Var]:
    """Interpolate features with normalized inverse Euclidean 8-NN weights.

    Set ``validate_finite=False`` only when the full batch has already passed
    an input-boundary finite-value check.
    """

    if not isinstance(validate_finite, bool):
        raise ValueError("validate_finite must be a bool")
    query_points = _validate_batched_points(
        query, "query", require_finite=validate_finite
    )
    reference_points = _validate_batched_points(
        reference, "reference", require_finite=validate_finite
    )
    features = _validate_batched_points(
        reference_features,
        "reference_features",
        require_xyz=False,
        require_finite=validate_finite,
    )
    if (
        features.shape[0] != reference_points.shape[0]
        or features.shape[1] != reference_points.shape[1]
    ):
        raise ValueError(
            "reference_features must match reference batch and point dimensions"
        )
    if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
        raise ValueError("epsilon must be a finite positive real scalar")
    epsilon_value = float(epsilon)
    if not math.isfinite(epsilon_value) or epsilon_value <= 0:
        raise ValueError("epsilon must be a finite positive real scalar")

    squared_distances, indices = knn(
        query_points,
        reference_points,
        k,
        query_block_size=query_block_size,
        reference_block_size=reference_block_size,
        validate_finite=False,
    )
    zero_mask = squared_distances == 0
    zero_choice, _ = zero_mask.int32().argmax(dim=-1)
    zero_weights = jt.nn.one_hot(
        zero_choice.int32(), int(squared_distances.shape[-1])
    ).cast(squared_distances.dtype)
    has_zero = zero_mask.sum(dim=-1, keepdims=True) > 0

    minimum_squared = jt.full_like(
        squared_distances, epsilon_value * epsilon_value
    )
    euclidean = jt.sqrt(jt.maximum(squared_distances, minimum_squared))
    inverse = 1.0 / euclidean
    normalized_inverse = inverse / inverse.sum(dim=-1, keepdims=True)
    weights = jt.where(has_zero, zero_weights, normalized_inverse)

    selected_features = _index_points_unchecked(features, indices)
    interpolated = (
        selected_features * weights.unsqueeze(-1)
    ).sum(dim=2)
    if return_details:
        return interpolated, squared_distances, indices, weights
    return interpolated

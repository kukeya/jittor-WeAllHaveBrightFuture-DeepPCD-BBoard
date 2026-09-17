"""Deterministic batched farthest-point sampling in pure Jittor."""

from __future__ import annotations

from numbers import Integral

import jittor as jt

from .indexing import (
    _index_points_unchecked,
    _validate_batched_points,
    _validate_knn_coordinate_range,
)
from .knn import _positive_integer


FPS_BACKENDS = ("auto", "reference", "cuda")
# Enabled after synchronized, materialized parity/latency evidence across every
# production PGD hierarchy size; see artifacts/progress/task07b_cuda_fps.
AUTO_CUDA_FPS_ENABLED = True


def farthest_point_sample(
    points: jt.Var,
    sample_count: int,
    *,
    start_index: int = 0,
    backend: str = "auto",
    validate_finite: bool = True,
) -> jt.Var:
    """Return unique FPS indices independently for every batch.

    Set ``validate_finite=False`` only after a batch-level finite-value
    preflight, avoiding a device synchronization inside stacked model stages.
    """

    if not isinstance(validate_finite, bool):
        raise ValueError("validate_finite must be a bool")
    coordinates = _validate_batched_points(
        points, "points", require_finite=validate_finite
    )
    if validate_finite:
        _validate_knn_coordinate_range(coordinates, "points")
    count = _positive_integer(sample_count, "sample_count")
    point_count = int(coordinates.shape[1])
    batch_size = int(coordinates.shape[0])
    if count > point_count:
        raise ValueError("sample_count must not exceed the point count")
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, Integral)
        or not 0 <= start_index < point_count
    ):
        raise ValueError("start_index is out of range")
    if not isinstance(backend, str) or backend not in FPS_BACKENDS:
        raise ValueError(
            f"backend must be one of {FPS_BACKENDS}, got {backend!r}"
        )
    selected_backend = backend
    if backend == "auto":
        can_use_fused_cuda = (
            AUTO_CUDA_FPS_ENABLED
            and jt.has_cuda
            and int(jt.flags.use_cuda)
            and str(coordinates.dtype) == "float32"
        )
        selected_backend = "cuda" if can_use_fused_cuda else "reference"
    if selected_backend == "cuda":
        from .cuda_fps import fused_cuda_fps

        return fused_cuda_fps(
            coordinates,
            count,
            start_index=int(start_index),
            validate_finite=False,
        )

    farthest = jt.full((batch_size,), int(start_index), dtype="int32").stop_grad()
    minimum_squared = jt.full(
        (batch_size, point_count), float("inf"), dtype=coordinates.dtype
    )
    selected_mask = jt.zeros(
        (batch_size, point_count), dtype=coordinates.dtype
    )
    selected_indices: list[jt.Var] = []

    for step in range(count):
        selected_indices.append(farthest)
        one_hot = jt.nn.one_hot(farthest, point_count).cast(coordinates.dtype)
        selected_mask = jt.maximum(selected_mask, one_hot)

        center = _index_points_unchecked(
            coordinates, farthest.reshape((batch_size, 1))
        )
        delta = coordinates - center
        squared = (delta * delta).sum(dim=-1)
        minimum_squared = jt.minimum(minimum_squared, squared)

        if step + 1 < count:
            candidate_scores = jt.where(
                selected_mask > 0,
                jt.full_like(minimum_squared, -1.0),
                minimum_squared,
            )
            farthest, _ = candidate_scores.argmax(dim=1)
            farthest = farthest.int32().stop_grad()

    return jt.stack(selected_indices, dim=1).int32().stop_grad()

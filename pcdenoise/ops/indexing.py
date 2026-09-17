"""Batch-preserving Jittor indexing without per-batch Python loops."""

from __future__ import annotations

from math import prod

import jittor as jt


_FLOAT_DTYPES = {"float16", "float32", "float64", "bfloat16"}
_INDEX_DTYPES = {"int32", "int64"}
_KNN_SAFE_ABSOLUTE_LIMITS = {
    # In three dimensions, |x| <= 64 bounds the worst float16 squared
    # distance by 3 * (2 * 64)^2 = 49152 < 65504.
    "float16": 64.0,
    # Candidate search is float32 for every other supported dtype. This
    # conservative bound leaves ample headroom below FLT_MAX:
    # 3 * (2 * 1e18)^2 = 1.2e37 < 3.4e38.
    "float32": 1.0e18,
    "float64": 1.0e18,
    "bfloat16": 1.0e18,
}


def _require_var(value: object, name: str) -> jt.Var:
    if not isinstance(value, jt.Var):
        raise TypeError(f"{name} must be a jittor.Var")
    return value


def _validate_batched_points(
    values: object,
    name: str,
    *,
    require_xyz: bool = True,
    require_finite: bool = True,
) -> jt.Var:
    points = _require_var(values, name)
    if points.ndim != 3:
        raise ValueError(f"{name} must have shape (B, N, C)")
    if points.shape[0] <= 0 or points.shape[1] <= 0 or points.shape[2] <= 0:
        raise ValueError(f"{name} dimensions must be non-empty")
    if require_xyz and points.shape[2] != 3:
        raise ValueError(f"{name} must have three coordinate channels")
    if str(points.dtype) not in _FLOAT_DTYPES:
        raise ValueError(f"{name} must have a floating dtype")
    if require_finite:
        finite_values = (
            points
            if str(points.dtype) == "float32"
            else points.float32()
        )
        if not bool(jt.isfinite(finite_values).all().item()):
            raise ValueError(f"{name} coordinates/features must be finite")
    return points


def _validate_knn_coordinate_range(points: jt.Var, name: str) -> None:
    """Reject finite coordinates whose squared distance could overflow.

    KNN callers operate on normalized point clouds. This explicit guard keeps
    the safe public path fail-closed even when individually finite coordinates
    would overflow subtraction, squaring, or three-channel accumulation.
    """

    limit = _KNN_SAFE_ABSOLUTE_LIMITS[str(points.dtype)]
    maximum_absolute = float(jt.abs(points.float32()).max().item())
    if maximum_absolute > limit:
        raise OverflowError(
            f"{name} coordinates exceed the safe KNN magnitude {limit:g}; "
            "normalize point coordinates before KNN"
        )


def _index_points_impl(
    points: jt.Var,
    indices: jt.Var,
    *,
    validate_range: bool,
) -> jt.Var:
    values = _require_var(points, "points")
    selected = _require_var(indices, "indices")
    if values.ndim < 3:
        raise ValueError("points must have shape (B, N, ...)")
    if values.shape[0] <= 0 or values.shape[1] <= 0:
        raise ValueError("points batch and point dimensions must be non-empty")
    if selected.ndim < 2:
        raise ValueError("indices must have shape (B, ...)")
    if selected.shape[0] != values.shape[0]:
        raise ValueError("points and indices batch dimensions must match")
    if str(selected.dtype) not in _INDEX_DTYPES:
        raise ValueError("indices must have dtype int32 or int64")

    batch_size, point_count = int(values.shape[0]), int(values.shape[1])
    trailing_shape = tuple(int(size) for size in values.shape[2:])
    trailing_size = int(prod(trailing_shape))
    index_shape = tuple(int(size) for size in selected.shape)
    selected_count = int(prod(index_shape[1:]))

    if validate_range and selected_count:
        minimum = int(selected.min().item())
        maximum = int(selected.max().item())
        if minimum < 0 or maximum >= point_count:
            raise IndexError(
                "indices are outside the valid point range "
                f"[0, {point_count}): min={minimum}, max={maximum}"
            )

    offset_shape = (batch_size,) + (1,) * (selected.ndim - 1)
    batch_offsets = (
        jt.arange(batch_size, dtype="int64").reshape(offset_shape)
        * point_count
    )
    flat_indices = (
        selected.int64() + batch_offsets
    ).reshape((-1,))
    flat_points = values.reshape((batch_size * point_count, trailing_size))
    gathered = flat_points[flat_indices]
    return gathered.reshape(index_shape + trailing_shape)


def _index_points_unchecked(points: jt.Var, indices: jt.Var) -> jt.Var:
    """Gather trusted internal indices without a device synchronization."""

    return _index_points_impl(points, indices, validate_range=False)


def index_points(points: jt.Var, indices: jt.Var) -> jt.Var:
    """Safely gather ``points[b, indices[b], ...]`` for every batch at once.

    Public calls validate index bounds before gathering. Internal operators use
    a private unchecked path only for indices they generated themselves. Batch
    offsets are always int64, preventing int32 overflow in flattened ``B*N``
    addressing.
    """

    return _index_points_impl(points, indices, validate_range=True)

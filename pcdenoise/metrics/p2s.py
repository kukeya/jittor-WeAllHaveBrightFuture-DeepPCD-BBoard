"""Exact organizer-style point-to-mesh-surface distance."""

from __future__ import annotations

import warnings

import numpy as np

from .chamfer import (
    DEGENERATE_SCALE_THRESHOLD,
    _reference_center_scale,
    as_float64_points,
)

try:
    import point_cloud_utils as pcu
except ImportError:  # pragma: no cover - the environment contract installs it.
    pcu = None


def _validated_faces(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    values = np.asarray(faces)
    # point-cloud-utils 0.34.0 squeezes a one-face OBJ to shape (3,).
    if values.ndim == 1 and values.shape == (3,):
        values = values.reshape(1, 3)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3), F > 0")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("faces must contain integer vertex indices")
    if (values < 0).any() or (values >= vertex_count).any():
        raise ValueError("face vertex index is out of range")
    if (values > np.iinfo(np.int32).max).any():
        raise ValueError("face vertex index exceeds point-cloud-utils range")
    values = values.astype(np.int32, copy=False)
    return np.ascontiguousarray(values)


def point_to_surface_distance(
    points: np.ndarray,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    *,
    normalize_reference: np.ndarray | None,
) -> float:
    """Return mean squared distance from points to the triangle mesh surface.

    No vertex-distance approximation is permitted in this strict evaluator.
    """

    if pcu is None:
        raise RuntimeError(
            "point-cloud-utils is required for exact P2S; "
            "vertex-only fallback is forbidden"
        )

    cloud = as_float64_points(points, "points")
    vertices = as_float64_points(mesh_vertices, "mesh_vertices")
    faces = _validated_faces(mesh_faces, len(vertices))

    if normalize_reference is not None:
        reference = as_float64_points(
            normalize_reference, "normalize_reference"
        )
        center, scale = _reference_center_scale(reference)
        # This edge behavior is copied from the organizer evaluator.
        if scale < DEGENERATE_SCALE_THRESHOLD:
            return 0.0
        cloud = (cloud - center) / scale
        vertices = (vertices - center) / scale

    # The official evaluator explicitly passes float32 geometry to PCU.
    # point-cloud-utils 0.34.0 squeezes a one-point query and dispatches it as
    # a single 3-vector, producing a scalar with incorrect geometry semantics.
    # Duplicating that one query keeps the batched API path; only the first
    # (identical) result is retained. Competition clouds have 50K points, so
    # this is a defensive small-fixture correction with no full-set effect.
    query_cloud = cloud
    single_query = len(query_cloud) == 1
    if single_query:
        query_cloud = np.repeat(query_cloud, 2, axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module="point_cloud_utils",
        )
        distances, _, _ = pcu.closest_points_on_mesh(
            np.ascontiguousarray(query_cloud.astype(np.float32)),
            np.ascontiguousarray(vertices.astype(np.float32)),
            faces,
        )
    distances = np.asarray(distances).reshape(-1)
    if single_query:
        distances = distances[:1]
    if distances.shape != (len(cloud),) or not np.isfinite(distances).all():
        raise RuntimeError("point-cloud-utils returned invalid P2S distances")
    result = float(np.square(distances).mean())
    if not np.isfinite(result) or result < 0.0:
        raise RuntimeError(f"P2S distance is invalid: {result}")
    return result

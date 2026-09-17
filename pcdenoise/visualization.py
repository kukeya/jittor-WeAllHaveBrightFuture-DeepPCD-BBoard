"""Host-side point-cloud preparation for TensorBoard visualization."""

from __future__ import annotations

import numpy as np


DEFAULT_POINT_COLOR = np.asarray([64, 160, 255], dtype=np.uint8)


def to_host_array(value: object, *, name: str = "array") -> np.ndarray:
    """Detach an array-like value to NumPy without importing another framework.

    Jittor variables expose ``.numpy()``; invoking it here synchronizes and
    copies the visualization value to the host, outside the differentiable
    training graph.
    """

    if isinstance(value, np.ndarray):
        array = value
    else:
        numpy_method = getattr(value, "numpy", None)
        if callable(numpy_method):
            array = numpy_method()
        else:
            array = value
    result = np.asarray(array)
    if not np.issubdtype(result.dtype, np.number) or np.issubdtype(
        result.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must be real numeric")
    return result


def prepare_point_cloud(
    points: object,
    *,
    colors: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return TensorBoard mesh-plugin vertices/colors with batch dimension.

    Points may have shape ``(N, 3)`` or ``(B, N, 3)``. Vertices are emitted as
    finite float32 and colors as uint8 RGB in ``[0, 255]``.
    """

    vertices = to_host_array(points, name="points")
    if vertices.ndim == 2:
        if vertices.shape[1:] != (3,):
            raise ValueError("points must have shape (N, 3) or (B, N, 3)")
        vertices = vertices[None, ...]
    elif vertices.ndim != 3 or vertices.shape[2] != 3:
        raise ValueError("points must have shape (N, 3) or (B, N, 3)")
    if vertices.shape[0] == 0 or vertices.shape[1] == 0:
        raise ValueError("point cloud batch and point count must be positive")
    with np.errstate(over="ignore", invalid="ignore"):
        vertices = vertices.astype(np.float32, copy=False)
    if not np.isfinite(vertices).all():
        raise ValueError("points must remain finite as float32")
    vertices = np.ascontiguousarray(vertices)

    if colors is None:
        color_array = np.broadcast_to(
            DEFAULT_POINT_COLOR,
            vertices.shape,
        ).copy()
    else:
        color_array = to_host_array(colors, name="colors")
        if color_array.ndim == 2:
            color_array = color_array[None, ...]
        if color_array.shape != vertices.shape:
            raise ValueError("colors must have the same point shape as points")
        if not np.isfinite(color_array).all():
            raise ValueError("colors must contain only finite values")
        if (color_array < 0).any() or (color_array > 255).any():
            raise ValueError("colors must lie in [0, 255]")
        if not np.equal(color_array, np.floor(color_array)).all():
            raise ValueError("colors must contain integer RGB values")
        color_array = color_array.astype(np.uint8, copy=False)
    return vertices, np.ascontiguousarray(color_array)

"""Deterministic OBJ loading and mesh-surface sampling utilities.

This module is deliberately NumPy-only.  It prepares clean training targets;
it is not part of the differentiable Jittor model.
"""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import TextIO

import numpy as np


AREA_SURFACE_PROFILE = "area_surface_v1"
STARTER_VERTEX_MIX_PROFILE = "starter_vertex_mix_v1"
PAIRED_VERTEX_REPLACEMENT_PROFILE = "paired_vertex_replacement_v1"
STARTER_FACE_VERTEX_MIX_PROFILE = "starter_face_vertex_mix_v2"
PAIRED_FACE_VERTEX_REPLACEMENT_PROFILE = (
    "paired_face_vertex_replacement_v2"
)
MESH_SAMPLING_PROFILES = (
    AREA_SURFACE_PROFILE,
    STARTER_VERTEX_MIX_PROFILE,
    PAIRED_VERTEX_REPLACEMENT_PROFILE,
    STARTER_FACE_VERTEX_MIX_PROFILE,
    PAIRED_FACE_VERTEX_REPLACEMENT_PROFILE,
)
_FACE_REFERENCED_VERTEX_PROFILES = frozenset(
    (
        STARTER_FACE_VERTEX_MIX_PROFILE,
        PAIRED_FACE_VERTEX_REPLACEMENT_PROFILE,
    )
)
_PAIRED_REPLACEMENT_PROFILES = frozenset(
    (
        PAIRED_VERTEX_REPLACEMENT_PROFILE,
        PAIRED_FACE_VERTEX_REPLACEMENT_PROFILE,
    )
)
_MESH_SAMPLING_STREAM_DOMAIN = b"pcdenoise:mesh_sampling_stream_v1\0"


def _as_points(values: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(values)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"{name} must have finite shape (N, 3), N > 0")
    if not np.issubdtype(points.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    points = points.astype(np.float32, copy=False)
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite values")
    return points


@dataclass(frozen=True)
class TriangleMesh:
    """A validated triangle mesh with canonical competition dtypes."""

    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        vertices = _as_points(self.vertices, "vertices").copy()
        raw_faces = np.asarray(self.faces)
        if (
            raw_faces.ndim != 2
            or raw_faces.shape[1] != 3
            or raw_faces.shape[0] == 0
        ):
            raise ValueError("faces must have shape (F, 3), F > 0")
        if not np.issubdtype(raw_faces.dtype, np.integer):
            raise ValueError("faces must contain integer vertex indices")
        faces = raw_faces.astype(np.int64, copy=True)
        if (faces < 0).any() or (faces >= len(vertices)).any():
            raise ValueError("face vertex index is out of range")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)


def _positive_surface_geometry(
    mesh: TriangleMesh,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return triangle geometry after checking positive finite total area."""

    triangles = mesh.vertices[mesh.faces].astype(np.float64)
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = float(areas.sum())
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise ValueError("mesh must have positive finite surface area")
    return triangles, areas, total_area


def _positive_area_vertex_indices(mesh: TriangleMesh) -> np.ndarray:
    """Return sorted vertices belonging to at least one nondegenerate face."""

    _, areas, _ = _positive_surface_geometry(mesh)
    indices = np.unique(mesh.faces[areas > 0.0].reshape(-1))
    if indices.size == 0:
        raise RuntimeError("positive-area mesh has no eligible vertices")
    return np.ascontiguousarray(indices, dtype=np.int64)


def _resolve_obj_index(token: str, vertex_count: int, line_number: int) -> int:
    vertex_token = token.split("/", 1)[0]
    if not vertex_token:
        raise ValueError(
            f"missing vertex index in OBJ face at line {line_number}"
        )
    try:
        raw_index = int(vertex_token)
    except ValueError as error:
        raise ValueError(
            f"invalid vertex index {vertex_token!r} at line {line_number}"
        ) from error
    if raw_index == 0:
        raise ValueError(
            f"OBJ vertex indices are never zero (line {line_number})"
        )
    index = raw_index - 1 if raw_index > 0 else vertex_count + raw_index
    if not 0 <= index < vertex_count:
        raise ValueError(
            f"OBJ vertex index {raw_index} is out of range at line "
            f"{line_number}"
        )
    return index


def _load_obj_stream(stream: TextIO, source: str) -> TriangleMesh:
    """Parse one already-opened text stream without reopening its source."""

    vertices = []
    faces = []
    for line_number, raw_line in enumerate(stream, start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        record = fields[0]
        if record == "v":
            if len(fields) < 4:
                raise ValueError(
                    f"OBJ vertex needs three coordinates at line "
                    f"{line_number}"
                )
            try:
                vertex = [float(value) for value in fields[1:4]]
            except ValueError as error:
                raise ValueError(
                    f"invalid OBJ vertex at line {line_number}"
                ) from error
            if not np.isfinite(vertex).all():
                raise ValueError(
                    f"non-finite OBJ vertex at line {line_number}"
                )
            vertices.append(vertex)
        elif record == "f":
            if len(fields) < 4:
                raise ValueError(
                    f"OBJ face needs at least three vertices at line "
                    f"{line_number}"
                )
            polygon = [
                _resolve_obj_index(token, len(vertices), line_number)
                for token in fields[1:]
            ]
            for offset in range(1, len(polygon) - 1):
                faces.append(
                    [polygon[0], polygon[offset], polygon[offset + 1]]
                )

    if not vertices:
        raise ValueError(f"OBJ has no vertices: {source}")
    if not faces:
        raise ValueError(f"OBJ has no faces: {source}")
    mesh = TriangleMesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
    )
    _positive_surface_geometry(mesh)
    return mesh


def load_obj(path: os.PathLike[str] | str) -> TriangleMesh:
    """Load OBJ vertices/faces and fan-triangulate polygonal faces.

    Texture/normal references in ``v/vt/vn`` and ``v//vn`` form are accepted.
    Geometry-irrelevant records such as ``vt``, ``vn``, ``usemtl`` and ``g``
    are ignored.
    """

    obj_path = Path(path)
    with obj_path.open("r", encoding="utf-8-sig") as stream:
        return _load_obj_stream(stream, str(obj_path))


def load_obj_bytes(payload: bytes) -> TriangleMesh:
    """Parse an immutable OBJ byte snapshot without reopening a path."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    text = payload.decode("utf-8-sig")
    return _load_obj_stream(io.StringIO(text), "<OBJ byte snapshot>")


def sample_surface(
    mesh: TriangleMesh,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample points from triangles in proportion to their surface area."""

    if (
        isinstance(num_points, bool)
        or not isinstance(num_points, (int, np.integer))
        or num_points <= 0
    ):
        raise ValueError("num_points must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")

    triangles, areas, total_area = _positive_surface_geometry(mesh)

    face_indices = rng.choice(
        len(triangles),
        size=int(num_points),
        replace=True,
        p=areas / total_area,
    )
    selected = triangles[face_indices]
    random_values = rng.random((int(num_points), 2))
    root = np.sqrt(random_values[:, 0])
    weights = np.column_stack(
        (
            1.0 - root,
            root * (1.0 - random_values[:, 1]),
            root * random_values[:, 1],
        )
    )
    points = np.einsum("ni,nij->nj", weights, selected)
    points = points.astype(np.float32)
    if not np.isfinite(points).all():
        raise ValueError("surface sampling produced non-finite points")
    return points


@dataclass(frozen=True)
class MeshPointSample:
    """One deterministic mesh sample and its component accounting.

    Vertex-mix samples keep selected mesh vertices first and area-sampled
    points second, matching the component order used by the official starter
    sampler.  Consumers may shuffle later with an independently specified
    policy; this primitive does not hide such a change.
    """

    points: np.ndarray
    profile: str
    requested_vertex_count: int
    vertex_indices: np.ndarray
    replacement_indices: np.ndarray
    surface_count: int
    area_baseline_points: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.area_baseline_points is None:
            return
        baseline = np.ascontiguousarray(
            _as_points(self.area_baseline_points, "area_baseline_points")
        )
        if baseline.shape != self.points.shape:
            raise ValueError(
                "area_baseline_points must match the sampled point shape"
            )
        if baseline.flags.writeable:
            baseline = baseline.copy()
            baseline.setflags(write=False)
        object.__setattr__(self, "area_baseline_points", baseline)

    @property
    def vertex_count(self) -> int:
        return int(self.vertex_indices.shape[0])


def _sampling_root_seed(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 <= int(value) < 2**64
    ):
        raise ValueError("seed must be an integer in [0, 2**64)")
    return int(value)


def _sampling_count(value: object, *, name: str, allow_zero: bool) -> int:
    minimum = 0 if allow_zero else 1
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < minimum
    ):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return int(value)


def _component_rng(seed: int, component: bytes) -> np.random.Generator:
    """Derive a stable RNG without consuming another component's stream."""

    digest = hashlib.sha256(
        _MESH_SAMPLING_STREAM_DOMAIN
        + seed.to_bytes(8, "little", signed=False)
        + len(component).to_bytes(8, "little", signed=False)
        + component
    ).digest()
    return np.random.default_rng(
        int.from_bytes(digest[:16], "little", signed=False)
    )


def _sample_surface_independent_streams(
    mesh: TriangleMesh,
    num_points: int,
    *,
    face_rng: np.random.Generator,
    barycentric_rng: np.random.Generator,
) -> np.ndarray:
    """Area sample with separate face and barycentric random streams."""

    if num_points == 0:
        return np.empty((0, 3), dtype=np.float32)
    triangles, areas, total_area = _positive_surface_geometry(mesh)
    face_indices = face_rng.choice(
        len(triangles),
        size=num_points,
        replace=True,
        p=areas / total_area,
    )
    selected = triangles[face_indices]
    random_values = barycentric_rng.random((num_points, 2))
    root = np.sqrt(random_values[:, 0])
    weights = np.column_stack(
        (
            1.0 - root,
            root * (1.0 - random_values[:, 1]),
            root * random_values[:, 1],
        )
    )
    points = np.einsum("ni,nij->nj", weights, selected).astype(np.float32)
    if not np.isfinite(points).all():
        raise ValueError("surface sampling produced non-finite points")
    return np.ascontiguousarray(points)


def sample_mesh_points(
    mesh: TriangleMesh,
    num_points: int,
    *,
    seed: int,
    profile: str = AREA_SURFACE_PROFILE,
    vertex_sample_count: int = 0,
) -> MeshPointSample:
    """Sample a mesh under an explicit, deterministic data profile.

    ``area_surface_v1`` is a compatibility wrapper around :func:`sample_surface`:
    for the same seed it returns exactly the legacy point array.

    ``starter_vertex_mix_v1`` samples mesh vertex *indices* without
    replacement, caps the requested count at the available vertex count, and
    fills the remainder from triangle area.  Vertex selection, face selection,
    and barycentric coordinates use independently derived random streams, so
    changing one component's count does not consume randomness from another.

    ``paired_vertex_replacement_v1`` first generates the complete legacy
    ``area_surface_v1`` array for the same seed.  Independently sampled mesh
    vertex indices then replace the same number of independently sampled point
    positions.  Every position not listed in ``replacement_indices`` remains
    bit-for-bit equal to the legacy array, making vertex supervision the only
    sampling-content intervention in a paired ablation.

    The corresponding ``*_face_*_v2`` profiles retain these two layouts but
    select only vertex indices referenced by at least one positive-area
    triangle.  OBJ files can contain unused ``v`` records far from every face,
    or vertices used only by degenerate faces; treating those records as clean
    targets corrupts both geometry and normalization.  The v1 profiles remain
    available so an already recorded experiment can still be interpreted
    against its original contract.
    """

    if not isinstance(mesh, TriangleMesh):
        raise TypeError("mesh must be TriangleMesh")
    total = _sampling_count(num_points, name="num_points", allow_zero=False)
    root_seed = _sampling_root_seed(seed)
    requested_vertices = _sampling_count(
        vertex_sample_count,
        name="vertex_sample_count",
        allow_zero=True,
    )
    if profile not in MESH_SAMPLING_PROFILES:
        raise ValueError(
            f"profile must be one of {MESH_SAMPLING_PROFILES}, "
            f"got {profile!r}"
        )

    if profile == AREA_SURFACE_PROFILE:
        if requested_vertices != 0:
            raise ValueError(
                "area_surface_v1 requires vertex_sample_count=0"
            )
        points = sample_surface(
            mesh,
            total,
            np.random.default_rng(root_seed),
        )
        return MeshPointSample(
            points=points,
            profile=profile,
            requested_vertex_count=0,
            vertex_indices=np.empty((0,), dtype=np.int64),
            replacement_indices=np.empty((0,), dtype=np.int64),
            surface_count=total,
        )

    if requested_vertices <= 0:
        raise ValueError(
            f"{profile} requires vertex_sample_count > 0"
        )
    if requested_vertices > total:
        raise ValueError("vertex_sample_count must not exceed num_points")

    if profile in _FACE_REFERENCED_VERTEX_PROFILES:
        eligible_vertex_indices = _positive_area_vertex_indices(mesh)
    else:
        eligible_vertex_indices = np.arange(
            len(mesh.vertices),
            dtype=np.int64,
        )
    eligible_vertex_indices = np.ascontiguousarray(
        eligible_vertex_indices,
        dtype=np.int64,
    )
    actual_vertex_count = min(
        requested_vertices,
        len(eligible_vertex_indices),
    )
    vertex_order = _component_rng(
        root_seed,
        b"vertex_indices",
    ).permutation(len(eligible_vertex_indices))
    vertex_indices = np.ascontiguousarray(
        eligible_vertex_indices[vertex_order[:actual_vertex_count]],
        dtype=np.int64,
    )
    surface_count = total - actual_vertex_count

    if profile in _PAIRED_REPLACEMENT_PROFILES:
        area_baseline = sample_surface(
            mesh,
            total,
            np.random.default_rng(root_seed),
        )
        points = area_baseline.copy()
        area_baseline.setflags(write=False)
        replacement_order = _component_rng(
            root_seed,
            b"replacement_positions",
        ).permutation(total)
        replacement_indices = np.ascontiguousarray(
            replacement_order[:actual_vertex_count],
            dtype=np.int64,
        )
        points[replacement_indices] = mesh.vertices[vertex_indices]
        points = np.ascontiguousarray(points, dtype=np.float32)
        if points.shape != (total, 3) or not np.isfinite(points).all():
            raise RuntimeError(
                "paired mesh sampling violated its array contract"
            )
        return MeshPointSample(
            points=points,
            profile=profile,
            requested_vertex_count=requested_vertices,
            vertex_indices=vertex_indices,
            replacement_indices=replacement_indices,
            surface_count=surface_count,
            area_baseline_points=area_baseline,
        )

    surface = _sample_surface_independent_streams(
        mesh,
        surface_count,
        face_rng=_component_rng(root_seed, b"surface_faces"),
        barycentric_rng=_component_rng(
            root_seed,
            b"surface_barycentric",
        ),
    )
    points = np.ascontiguousarray(
        np.concatenate(
            (mesh.vertices[vertex_indices], surface),
            axis=0,
        ),
        dtype=np.float32,
    )
    if points.shape != (total, 3) or not np.isfinite(points).all():
        raise RuntimeError("mixed mesh sampling violated its array contract")
    return MeshPointSample(
        points=points,
        profile=profile,
        requested_vertex_count=requested_vertices,
        vertex_indices=vertex_indices,
        replacement_indices=np.empty((0,), dtype=np.int64),
        surface_count=surface_count,
    )


@dataclass(frozen=True)
class UnitSphereTransform:
    """One reference-derived transform shared by every related geometry.

    ``center`` and ``scale`` remain float64 so they describe one internally
    consistent frame even when a large float32 coordinate offset makes the
    bbox midpoint itself unrepresentable in float32.  Point-array outputs stay
    float32 for the training/evaluation data contract.
    """

    center: np.ndarray
    scale: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        scale = float(self.scale)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("center must be a finite length-three vector")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be positive and finite")
        object.__setattr__(self, "center", center.copy())
        object.__setattr__(self, "scale", scale)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Apply this transform without refitting it to ``points``."""

        values = _as_points(points, "points")
        transformed = (
            values.astype(np.float64) - self.center.astype(np.float64)
        ) / self.scale
        result = transformed.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("normalization produced non-finite points")
        return result

    def restore(self, points: np.ndarray) -> np.ndarray:
        """Invert this transform."""

        values = _as_points(points, "points")
        restored = (
            values.astype(np.float64) * self.scale
            + self.center.astype(np.float64)
        )
        result = restored.astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("inverse normalization produced non-finite points")
        return result


def fit_unit_sphere(reference_points: np.ndarray) -> UnitSphereTransform:
    """Fit the official bbox-center/max-radius transform to one reference.

    The caller chooses the clean sampled surface or mesh vertices as the
    reference, then reuses the returned transform for noisy points,
    predictions, and mesh geometry.
    """

    reference = _as_points(reference_points, "reference_points").astype(
        np.float64
    )
    center = (reference.min(axis=0) + reference.max(axis=0)) * 0.5
    scale = float(np.linalg.norm(reference - center, axis=1).max())
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("reference geometry has degenerate unit-sphere scale")
    return UnitSphereTransform(center=center, scale=scale)

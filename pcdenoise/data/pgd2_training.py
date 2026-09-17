"""Deterministic paired-patch loading for README-defined PGD2 training.

PGD2 consumes the frozen PGD1 output as its input and the same-index clean
cloud as its target.  This module deliberately does not generate noise.  Every
epoch visits exactly ``patches_per_shape`` patches per shape, while the center
choices and flattened patch order are reproducible from ``(seed, epoch)`` and
can therefore be rebuilt exactly after a mid-epoch resume.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path, PurePosixPath

import numpy as np

from pcdenoise.data.mesh_dataset import UnitSphereTransform, fit_unit_sphere
from pcdenoise.data.pgd2_paired_cache import verify_pgd2_paired_cache


_UINT64_LIMIT = 2**64
_SHUFFLE_DOMAIN = b"pcdenoise:pgd2_epoch_shape_shuffle_v1\0"
_CENTER_DOMAIN = b"pcdenoise:pgd2_epoch_centers_v1\0"
_PATCH_ORDER_DOMAIN = b"pcdenoise:pgd2_epoch_patch_order_v1\0"


def _positive_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _uint64(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < 0
        or int(value) >= _UINT64_LIMIT
    ):
        raise ValueError(
            f"{name} must be a nonnegative integer smaller than 2**64"
        )
    return int(value)


def _snapshot_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"input is not a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _strict_relative_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} must be a strict POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{name} must be a strict POSIX relative path")
    if path.as_posix() != value:
        raise ValueError(f"{name} must be a canonical POSIX relative path")
    return value


def _sample_ids_sha256(sample_ids: tuple[str, ...]) -> str:
    return hashlib.sha256(
        "".join(f"{sample_id}\n" for sample_id in sample_ids).encode("ascii")
    ).hexdigest()


def _stream_seed(
    domain: bytes,
    *,
    seed: int,
    epoch: int,
    sample_id: str | None = None,
) -> int:
    payload = (
        domain
        + seed.to_bytes(8, "little", signed=False)
        + epoch.to_bytes(8, "little", signed=False)
    )
    if sample_id is not None:
        encoded = sample_id.encode("ascii")
        payload += len(encoded).to_bytes(8, "little", signed=False) + encoded
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "little")


def _readonly_float32(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("PGD2 preprocessing produced non-finite points")
    result.setflags(write=False)
    return result


def _readonly_int64(values: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(values, dtype=np.int64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PGD2PairedSample:
    """One provenance-bound whole-cloud PGD2 input/target pair."""

    sample_id: str
    pgd1_input_path: Path
    clean_target_path: Path
    pgd1_input_sha256: str
    clean_target_sha256: str
    pgd1_input_file_bytes: int
    clean_target_file_bytes: int
    point_count: int


@dataclass(frozen=True)
class PGD2PairedDataset:
    """An exact snapshot of one completed paired-cache manifest."""

    root: Path
    manifest_sha256: str
    content_sha256: str
    pgd1_checkpoint_sha256: str
    pgd1_config_sha256: str
    pgd1_checkpoint_step: int | None
    point_count: int
    samples: tuple[PGD2PairedSample, ...]
    sample_ids_sha256: str

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sample.sample_id for sample in self.samples)


@dataclass(frozen=True)
class PGD2PatchRequest:
    """One center selection in a deterministic epoch plan."""

    sample_index: int
    sample_id: str
    patch_rank: int
    center_index: int


@dataclass(frozen=True)
class PGD2EpochPlan:
    """The complete batch-size-independent patch order for one epoch."""

    seed: int
    epoch: int
    patches_per_shape: int
    sample_count: int
    point_count: int
    sample_ids_sha256: str
    requests: tuple[PGD2PatchRequest, ...]

    @property
    def patch_count(self) -> int:
        return len(self.requests)

    def batch_count(self, batch_size: int) -> int:
        size = _positive_integer(batch_size, name="batch_size")
        return math.ceil(self.patch_count / size)


@dataclass(frozen=True)
class PGD2TrainingCloud:
    """One pair normalized by a transform fitted only to the PGD1 input."""

    sample_id: str
    pgd1_input: np.ndarray
    clean_target: np.ndarray
    transform: UnitSphereTransform


@dataclass(frozen=True)
class PGD2TrainingPatch:
    """One same-index PGD1-input/clean patch in centered normalized units."""

    request: PGD2PatchRequest
    pgd1_input: np.ndarray
    clean_target: np.ndarray
    indices: np.ndarray
    seed_point: np.ndarray
    cloud: PGD2TrainingCloud

    @property
    def transform(self) -> UnitSphereTransform:
        """Return the whole-cloud transform needed to restore PGD2 output."""

        return self.cloud.transform


@dataclass(frozen=True)
class PGD2TrainingBatch:
    """One optimization batch; ``batch_size`` counts patches, not shapes."""

    pgd1_input: np.ndarray
    clean_target: np.ndarray
    indices: np.ndarray
    seed_points: np.ndarray
    sample_ids: tuple[str, ...]
    patch_ranks: tuple[int, ...]
    center_indices: tuple[int, ...]
    transforms: tuple[UnitSphereTransform, ...]
    requests: tuple[PGD2PatchRequest, ...]
    epoch: int
    batch_index: int
    epoch_patch_count: int
    epoch_batch_count: int
    prefix_patch_count: int


def load_pgd2_paired_dataset(
    cache_dir: os.PathLike[str] | str,
    *,
    expected_content_sha256: str,
    expected_pgd1_checkpoint_sha256: str,
    expected_pgd1_config_sha256: str,
    expected_point_count: int = 50_000,
) -> PGD2PairedDataset:
    """Fully verify one paired cache and return immutable sample descriptors.

    Every payload is verified here.  Batch reads repeat each file's declared
    byte-count and SHA256 check so a later replacement cannot silently alter a
    resumed trajectory.
    """

    root = Path(cache_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    point_count = _positive_integer(
        expected_point_count,
        name="expected_point_count",
    )
    manifest_path = root / "manifest.json"
    before = _snapshot_regular(manifest_path)
    manifest = verify_pgd2_paired_cache(
        root,
        expected_content_sha256=expected_content_sha256,
        expected_pgd1_checkpoint_sha256=(
            expected_pgd1_checkpoint_sha256
        ),
        expected_pgd1_config_sha256=expected_pgd1_config_sha256,
        verify_files=True,
    )
    after = _snapshot_regular(manifest_path)
    if before != after:
        raise RuntimeError("PGD2 paired manifest changed during verification")
    try:
        snapshot = json.loads(before.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid PGD2 paired manifest snapshot") from error
    if snapshot != manifest:
        raise RuntimeError("verified PGD2 paired manifest snapshot differs")
    if manifest["point_count"] != point_count:
        raise ValueError("PGD2 paired point_count does not match expected value")

    descriptors = []
    for sample_id, record in zip(manifest["sample_ids"], manifest["samples"]):
        input_record = record["pgd2_input"]
        target_record = record["clean_target"]
        input_relative = _strict_relative_path(
            input_record["relative_path"],
            name="PGD2 input relative_path",
        )
        target_relative = _strict_relative_path(
            target_record["relative_path"],
            name="clean target relative_path",
        )
        descriptors.append(
            PGD2PairedSample(
                sample_id=sample_id,
                pgd1_input_path=root / input_relative,
                clean_target_path=root / target_relative,
                pgd1_input_sha256=input_record["sha256"],
                clean_target_sha256=target_record["sha256"],
                pgd1_input_file_bytes=int(input_record["file_bytes"]),
                clean_target_file_bytes=int(target_record["file_bytes"]),
                point_count=point_count,
            )
        )
    samples = tuple(descriptors)
    sample_ids = tuple(sample.sample_id for sample in samples)
    pgd1 = manifest["pgd1"]
    return PGD2PairedDataset(
        root=root,
        manifest_sha256=hashlib.sha256(before).hexdigest(),
        content_sha256=manifest["content_sha256"],
        pgd1_checkpoint_sha256=pgd1["checkpoint_sha256"],
        pgd1_config_sha256=pgd1["config_sha256"],
        pgd1_checkpoint_step=(
            int(pgd1["checkpoint_step"])
            if "checkpoint_step" in pgd1
            else None
        ),
        point_count=point_count,
        samples=samples,
        sample_ids_sha256=_sample_ids_sha256(sample_ids),
    )


def _load_bound_cloud(
    path: Path,
    *,
    expected_sha256: str,
    expected_file_bytes: int,
    expected_point_count: int,
    label: str,
) -> np.ndarray:
    payload = _snapshot_regular(path)
    if len(payload) != expected_file_bytes:
        raise ValueError(f"{label} file byte count mismatch")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch")
    try:
        values = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid {label} NPY: {path}") from error
    if (
        values.dtype != np.float32
        or values.shape != (expected_point_count, 3)
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            f"{label} must have finite shape "
            f"({expected_point_count}, 3) float32"
        )
    return _readonly_float32(values)


def load_pgd2_training_cloud(sample: PGD2PairedSample) -> PGD2TrainingCloud:
    """Load and normalize a pair using README ``noisy_max`` semantics."""

    if not isinstance(sample, PGD2PairedSample):
        raise TypeError("sample must be a PGD2PairedSample")
    pgd1_input = _load_bound_cloud(
        sample.pgd1_input_path,
        expected_sha256=sample.pgd1_input_sha256,
        expected_file_bytes=sample.pgd1_input_file_bytes,
        expected_point_count=sample.point_count,
        label="PGD1 input",
    )
    clean_target = _load_bound_cloud(
        sample.clean_target_path,
        expected_sha256=sample.clean_target_sha256,
        expected_file_bytes=sample.clean_target_file_bytes,
        expected_point_count=sample.point_count,
        label="clean target",
    )
    transform = fit_unit_sphere(pgd1_input)
    normalized_input = _readonly_float32(transform.apply(pgd1_input))
    normalized_target = _readonly_float32(transform.apply(clean_target))
    return PGD2TrainingCloud(
        sample_id=sample.sample_id,
        pgd1_input=normalized_input,
        clean_target=normalized_target,
        transform=transform,
    )


def build_pgd2_epoch_plan(
    dataset: PGD2PairedDataset,
    *,
    seed: int,
    epoch: int,
    patches_per_shape: int = 4,
) -> PGD2EpochPlan:
    """Build a 0-based, batch-size-independent deterministic epoch plan."""

    if not isinstance(dataset, PGD2PairedDataset):
        raise TypeError("dataset must be a PGD2PairedDataset")
    base_seed = _uint64(seed, name="seed")
    epoch_index = _uint64(epoch, name="epoch")
    patch_count = _positive_integer(
        patches_per_shape,
        name="patches_per_shape",
    )
    if patch_count > dataset.point_count:
        raise ValueError("patches_per_shape must not exceed point_count")

    shape_rng = np.random.default_rng(
        _stream_seed(
            _SHUFFLE_DOMAIN,
            seed=base_seed,
            epoch=epoch_index,
        )
    )
    shape_order = shape_rng.permutation(len(dataset.samples))
    requests = []
    for raw_sample_index in shape_order:
        sample_index = int(raw_sample_index)
        sample = dataset.samples[sample_index]
        center_rng = np.random.default_rng(
            _stream_seed(
                _CENTER_DOMAIN,
                seed=base_seed,
                epoch=epoch_index,
                sample_id=sample.sample_id,
            )
        )
        centers = center_rng.choice(
            dataset.point_count,
            size=patch_count,
            replace=False,
        )
        order_rng = np.random.default_rng(
            _stream_seed(
                _PATCH_ORDER_DOMAIN,
                seed=base_seed,
                epoch=epoch_index,
                sample_id=sample.sample_id,
            )
        )
        for raw_rank in order_rng.permutation(patch_count):
            patch_rank = int(raw_rank)
            requests.append(
                PGD2PatchRequest(
                    sample_index=sample_index,
                    sample_id=sample.sample_id,
                    patch_rank=patch_rank,
                    center_index=int(centers[patch_rank]),
                )
            )
    return PGD2EpochPlan(
        seed=base_seed,
        epoch=epoch_index,
        patches_per_shape=patch_count,
        sample_count=len(dataset.samples),
        point_count=dataset.point_count,
        sample_ids_sha256=dataset.sample_ids_sha256,
        requests=tuple(requests),
    )


def _validate_plan_dataset(
    dataset: PGD2PairedDataset,
    plan: PGD2EpochPlan,
) -> None:
    if not isinstance(dataset, PGD2PairedDataset):
        raise TypeError("dataset must be a PGD2PairedDataset")
    if not isinstance(plan, PGD2EpochPlan):
        raise TypeError("plan must be a PGD2EpochPlan")
    if (
        plan.sample_count != len(dataset.samples)
        or plan.point_count != dataset.point_count
        or plan.sample_ids_sha256 != dataset.sample_ids_sha256
        or plan.patch_count != len(dataset.samples) * plan.patches_per_shape
    ):
        raise ValueError("PGD2 epoch plan does not match the paired dataset")


def pgd2_epoch_batches(
    plan: PGD2EpochPlan,
    *,
    batch_size: int,
    batch_offset: int = 0,
) -> tuple[tuple[PGD2PatchRequest, ...], ...]:
    """Return full or resumed unpadded patch batches from an epoch plan."""

    if not isinstance(plan, PGD2EpochPlan):
        raise TypeError("plan must be a PGD2EpochPlan")
    size = _positive_integer(batch_size, name="batch_size")
    offset = _uint64(batch_offset, name="batch_offset")
    batch_count = plan.batch_count(size)
    if offset > batch_count:
        raise ValueError("batch_offset exceeds the epoch batch count")
    return tuple(
        plan.requests[start : min(start + size, plan.patch_count)]
        for start in range(offset * size, plan.patch_count, size)
    )


def _knn_indices(
    points: np.ndarray,
    *,
    center_index: int,
    patch_size: int,
) -> np.ndarray:
    point_count = len(points)
    center = _uint64(center_index, name="center_index")
    if center >= point_count:
        raise ValueError("center_index is out of range")
    count = _positive_integer(patch_size, name="patch_size")
    if count > point_count:
        raise ValueError("patch_size must not exceed point_count")
    delta = points.astype(np.float64) - points[center].astype(np.float64)
    squared = np.einsum("ij,ij->i", delta, delta)
    if count == point_count:
        candidates = np.arange(point_count, dtype=np.int64)
    else:
        candidates = np.argpartition(squared, count - 1)[:count]
    order = np.lexsort((candidates, squared[candidates]))
    return _readonly_int64(candidates[order])


def build_pgd2_training_patch(
    sample: PGD2PairedSample,
    request: PGD2PatchRequest,
    *,
    patch_size: int = 1000,
    cloud: PGD2TrainingCloud | None = None,
) -> PGD2TrainingPatch:
    """Build one KNN patch, using PGD1 input indices for both geometries."""

    if not isinstance(sample, PGD2PairedSample):
        raise TypeError("sample must be a PGD2PairedSample")
    if not isinstance(request, PGD2PatchRequest):
        raise TypeError("request must be a PGD2PatchRequest")
    if request.sample_id != sample.sample_id:
        raise ValueError("patch request sample_id does not match sample")
    prepared = load_pgd2_training_cloud(sample) if cloud is None else cloud
    if not isinstance(prepared, PGD2TrainingCloud):
        raise TypeError("cloud must be a PGD2TrainingCloud")
    if prepared.sample_id != sample.sample_id:
        raise ValueError("prepared cloud sample_id does not match sample")
    indices = _knn_indices(
        prepared.pgd1_input,
        center_index=request.center_index,
        patch_size=patch_size,
    )
    seed_point = _readonly_float32(
        prepared.pgd1_input[request.center_index].copy()
    )
    input_patch = _readonly_float32(
        prepared.pgd1_input[indices] - seed_point
    )
    target_patch = _readonly_float32(
        prepared.clean_target[indices] - seed_point
    )
    return PGD2TrainingPatch(
        request=request,
        pgd1_input=input_patch,
        clean_target=target_patch,
        indices=indices,
        seed_point=seed_point,
        cloud=prepared,
    )


def build_pgd2_training_batch(
    dataset: PGD2PairedDataset,
    plan: PGD2EpochPlan,
    *,
    batch_index: int,
    batch_size: int,
    patch_size: int = 1000,
) -> PGD2TrainingBatch:
    """Rebuild one exact batch directly from its epoch/batch cursor."""

    _validate_plan_dataset(dataset, plan)
    size = _positive_integer(batch_size, name="batch_size")
    selected_batch = _uint64(batch_index, name="batch_index")
    batch_count = plan.batch_count(size)
    if selected_batch >= batch_count:
        raise ValueError("batch_index must be smaller than epoch batch count")
    start = selected_batch * size
    requests = plan.requests[start : min(start + size, plan.patch_count)]

    clouds: dict[int, PGD2TrainingCloud] = {}
    patches = []
    for request in requests:
        if not 0 <= request.sample_index < len(dataset.samples):
            raise ValueError("patch request sample_index is out of range")
        sample = dataset.samples[request.sample_index]
        if request.sample_id != sample.sample_id:
            raise ValueError("patch request does not match dataset ordering")
        if request.sample_index not in clouds:
            clouds[request.sample_index] = load_pgd2_training_cloud(sample)
        patches.append(
            build_pgd2_training_patch(
                sample,
                request,
                patch_size=patch_size,
                cloud=clouds[request.sample_index],
            )
        )

    inputs = _readonly_float32(
        np.stack([patch.pgd1_input for patch in patches])
    )
    targets = _readonly_float32(
        np.stack([patch.clean_target for patch in patches])
    )
    indices = _readonly_int64(
        np.stack([patch.indices for patch in patches])
    )
    seed_points = _readonly_float32(
        np.stack([patch.seed_point for patch in patches])
    )
    return PGD2TrainingBatch(
        pgd1_input=inputs,
        clean_target=targets,
        indices=indices,
        seed_points=seed_points,
        sample_ids=tuple(patch.request.sample_id for patch in patches),
        patch_ranks=tuple(patch.request.patch_rank for patch in patches),
        center_indices=tuple(
            patch.request.center_index for patch in patches
        ),
        transforms=tuple(patch.transform for patch in patches),
        requests=requests,
        epoch=plan.epoch,
        batch_index=selected_batch,
        epoch_patch_count=plan.patch_count,
        epoch_batch_count=batch_count,
        prefix_patch_count=min((selected_batch + 1) * size, plan.patch_count),
    )


__all__ = [
    "PGD2EpochPlan",
    "PGD2PairedDataset",
    "PGD2PairedSample",
    "PGD2PatchRequest",
    "PGD2TrainingBatch",
    "PGD2TrainingCloud",
    "PGD2TrainingPatch",
    "build_pgd2_epoch_plan",
    "build_pgd2_training_batch",
    "build_pgd2_training_patch",
    "load_pgd2_paired_dataset",
    "load_pgd2_training_cloud",
    "pgd2_epoch_batches",
]

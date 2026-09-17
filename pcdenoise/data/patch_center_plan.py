"""Deterministic cross-visit FPS centers for cached training clouds.

The plan is deliberately stored outside the immutable surface cache.  A v1
plan has ``(shape, center)`` layout; a v2 plan binds an independent
``(shape, view, center)`` row to every nested cache view.  Keeping this
artifact separate preserves existing surface-cache and checkpoint contracts.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .archive import (
    MODEL_RE,
    SYNSET_RE,
    _cleanup_owned_stage,
    _create_owned_stage,
    _publish_owned_stage,
)


PATCH_CENTER_PLAN_FORMAT = "pcdenoise_patch_center_plan_v1"
PATCH_CENTER_PLAN_VERSION = 1
PATCH_CENTER_PLAN_V2_FORMAT = "pcdenoise_patch_center_plan_v2"
PATCH_CENTER_PLAN_V2_VERSION = 2
PATCH_CENTER_MODE = "fps_epoch_v1"
_ALGORITHM = (
    "float64_iterative_fps_start0_argmax_low_index_tie_v1"
)
_MANIFEST_KEYS = frozenset(
    {
        "format",
        "format_version",
        "mode",
        "algorithm",
        "shape_count",
        "sample_ids",
        "sample_ids_sha256",
        "point_count",
        "center_count",
        "start_index",
        "tie_break",
        "array_contract",
        "source_cache_content_sha256",
        "centers_sha256",
        "centers_file_bytes",
        "source_sha256",
        "content_sha256",
    }
)
_V2_MANIFEST_KEYS = frozenset(
    set(_MANIFEST_KEYS) | {"view_count", "source_views_sha256"}
)


@dataclass(frozen=True)
class PatchCenterPlan:
    """Verified read-only FPS center table and its provenance."""

    indices: np.ndarray
    sample_ids: tuple[str, ...]
    center_count: int
    point_count: int
    content_sha256: str
    source_cache_content_sha256: str
    root: Path
    view_count: int = 1
    source_views_sha256: str | None = None


def _positive_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _sample_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("sample IDs must be strings")
    parts = value.split("/")
    if (
        len(parts) != 2
        or SYNSET_RE.fullmatch(parts[0]) is None
        or MODEL_RE.fullmatch(parts[1]) is None
    ):
        raise ValueError(
            "sample IDs must have form <8-digit synset>/<28-32 hex model>"
        )
    return value


def _sample_ids(values: object) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("sample_ids must be a nonempty list")
    result = tuple(_sample_id(value) for value in values)
    if tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise ValueError("sample_ids must be unique and sorted")
    return result


def _sample_ids_digest(values: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in values).encode("ascii")
    ).hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _manifest_digest(manifest: Mapping[str, object]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("content_sha256", None)
    return _canonical_digest(unsigned)


def _file_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _source_sha256() -> str:
    return _file_sha256(Path(__file__).resolve())[1]


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def _surface_cache_contract(
    cache_root: Path,
) -> tuple[
    tuple[str, ...],
    int,
    str,
    tuple[tuple[str, int, str], ...],
]:
    if not cache_root.is_dir():
        raise FileNotFoundError(cache_root)
    manifest = _read_json(
        cache_root / "manifest.json",
        label="surface-cache manifest",
    )
    if (
        manifest.get("format") != "pcdenoise_train_surface_cache_v1"
        or manifest.get("format_version") != 1
    ):
        raise ValueError("surface-cache format/version is not supported")
    sample_ids = _sample_ids(manifest.get("shape_ids"))
    shape_count = _positive_integer(
        manifest.get("shape_count"),
        name="surface-cache shape_count",
    )
    point_count = _positive_integer(
        manifest.get("num_points"),
        name="surface-cache num_points",
    )
    if shape_count != len(sample_ids):
        raise ValueError("surface-cache shape_count does not match shape_ids")
    content_sha256 = _sha256(
        manifest.get("content_sha256"),
        name="surface-cache content_sha256",
    )
    if _manifest_digest(manifest) != content_sha256:
        raise ValueError("surface-cache manifest content hash does not match")
    raw_records = manifest.get("samples")
    if not isinstance(raw_records, list) or len(raw_records) != len(sample_ids):
        raise ValueError("surface-cache samples do not match shape_ids")
    if manifest.get("samples_sha256") != _canonical_digest(raw_records):
        raise ValueError("surface-cache samples hash does not match")
    records = []
    for sample_id, raw_record in zip(sample_ids, raw_records):
        if not isinstance(raw_record, Mapping):
            raise ValueError("surface-cache sample records must be mappings")
        expected_relative = (
            Path("shapenet") / sample_id / "clean.npy"
        ).as_posix()
        relative_path = raw_record.get("relative_path")
        file_bytes = _positive_integer(
            raw_record.get("clean_file_bytes"),
            name="surface-cache clean_file_bytes",
        )
        clean_sha256 = _sha256(
            raw_record.get("clean_sha256"),
            name="surface-cache clean_sha256",
        )
        if (
            raw_record.get("shape_id") != sample_id
            or relative_path != expected_relative
        ):
            raise ValueError(
                "surface-cache sample record path/order does not match"
            )
        records.append((expected_relative, file_bytes, clean_sha256))
    return sample_ids, point_count, content_sha256, tuple(records)


def _surface_cache_contract_any(
    cache_root: Path,
) -> tuple[
    tuple[str, ...],
    int,
    str,
    int,
    int,
    tuple[tuple[tuple[str, int, str], ...], ...],
    str,
]:
    """Read v1 as one view or validate every nested v2 view record."""

    manifest = _read_json(
        cache_root / "manifest.json",
        label="surface-cache manifest",
    )
    if (
        manifest.get("format") == "pcdenoise_train_surface_cache_v1"
        and manifest.get("format_version") == 1
    ):
        sample_ids, point_count, content_sha256, records = (
            _surface_cache_contract(cache_root)
        )
        nested = tuple((record,) for record in records)
        return (
            sample_ids,
            point_count,
            content_sha256,
            1,
            1,
            nested,
            _canonical_digest(
                [
                    {
                        "shape_id": sample_id,
                        "views": [
                            {
                                "view_id": 0,
                                "relative_path": record[0],
                                "clean_file_bytes": record[1],
                                "clean_sha256": record[2],
                            }
                        ],
                    }
                    for sample_id, record in zip(sample_ids, records)
                ]
            ),
        )
    if (
        manifest.get("format") != "pcdenoise_train_surface_cache_v2"
        or manifest.get("format_version") != 2
    ):
        raise ValueError("surface-cache format/version is not supported")

    sample_ids = _sample_ids(manifest.get("shape_ids"))
    shape_count = _positive_integer(
        manifest.get("shape_count"),
        name="surface-cache shape_count",
    )
    point_count = _positive_integer(
        manifest.get("num_points"),
        name="surface-cache num_points",
    )
    view_count = _positive_integer(
        manifest.get("view_count"),
        name="surface-cache view_count",
    )
    if view_count not in (1, 2):
        raise ValueError("surface-cache view_count must be 1 or 2")
    if shape_count != len(sample_ids):
        raise ValueError("surface-cache shape_count does not match shape_ids")
    content_sha256 = _sha256(
        manifest.get("content_sha256"),
        name="surface-cache content_sha256",
    )
    if _manifest_digest(manifest) != content_sha256:
        raise ValueError("surface-cache manifest content hash does not match")
    raw_records = manifest.get("samples")
    if not isinstance(raw_records, list) or len(raw_records) != len(sample_ids):
        raise ValueError("surface-cache samples do not match shape_ids")
    if manifest.get("samples_sha256") != _canonical_digest(raw_records):
        raise ValueError("surface-cache samples hash does not match")

    nested_records = []
    source_views = []
    for sample_id, raw_record in zip(sample_ids, raw_records):
        if not isinstance(raw_record, Mapping):
            raise ValueError("surface-cache sample records must be mappings")
        if raw_record.get("shape_id") != sample_id:
            raise ValueError("surface-cache sample order does not match")
        raw_views = raw_record.get("views")
        if not isinstance(raw_views, list) or len(raw_views) != view_count:
            raise ValueError("surface-cache view records do not match")
        records = []
        digest_views = []
        for expected_view_id, raw_view in enumerate(raw_views):
            if not isinstance(raw_view, Mapping):
                raise ValueError("surface-cache views must be mappings")
            if raw_view.get("view_id") != expected_view_id:
                raise ValueError("surface-cache view order does not match")
            filename = (
                "clean.npy"
                if expected_view_id == 0
                else f"clean_view_{expected_view_id:03d}.npy"
            )
            expected_relative = (
                Path("shapenet") / sample_id / filename
            ).as_posix()
            relative_path = raw_view.get("relative_path")
            file_bytes = _positive_integer(
                raw_view.get("clean_file_bytes"),
                name="surface-cache clean_file_bytes",
            )
            clean_sha256 = _sha256(
                raw_view.get("clean_sha256"),
                name="surface-cache clean_sha256",
            )
            if relative_path != expected_relative:
                raise ValueError("surface-cache view path does not match")
            records.append((expected_relative, file_bytes, clean_sha256))
            digest_views.append(
                {
                    "view_id": expected_view_id,
                    "relative_path": expected_relative,
                    "clean_file_bytes": file_bytes,
                    "clean_sha256": clean_sha256,
                }
            )
        nested_records.append(tuple(records))
        source_views.append(
            {"shape_id": sample_id, "views": digest_views}
        )
    return (
        sample_ids,
        point_count,
        content_sha256,
        2,
        view_count,
        tuple(nested_records),
        _canonical_digest(source_views),
    )


def _load_clean(
    path: Path,
    *,
    point_count: int,
    expected_file_bytes: int,
    expected_sha256: str,
) -> np.ndarray:
    try:
        payload = path.read_bytes()
        if (
            len(payload) != expected_file_bytes
            or hashlib.sha256(payload).hexdigest() != expected_sha256
        ):
            raise ValueError(
                f"clean cache file size/SHA does not match manifest: {path}"
            )
        values = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        if "size/SHA" in str(error):
            raise
        raise ValueError(f"invalid clean cache array: {path}") from error
    if values.dtype != np.float32 or values.shape != (point_count, 3):
        raise ValueError(
            "clean cache array must have declared shape and float32 dtype: "
            f"{path}"
        )
    result = np.ascontiguousarray(values)
    if not np.isfinite(result).all():
        raise ValueError(f"clean cache array must be finite: {path}")
    return result


def _farthest_point_indices(
    points: np.ndarray,
    count: int,
) -> np.ndarray:
    """Match inference FPS exactly without importing Jittor into the builder."""

    point_count = len(points)
    if count > point_count:
        raise ValueError("center_count must not exceed the point count")
    coordinates = points.astype(np.float64)
    output = np.empty(count, dtype=np.int32)
    minimum_squared = np.full(point_count, np.inf, dtype=np.float64)
    selected = np.zeros(point_count, dtype=bool)
    current = 0
    for step in range(count):
        output[step] = np.int32(current)
        selected[current] = True
        delta = coordinates - coordinates[current]
        squared = np.einsum("ij,ij->i", delta, delta)
        minimum_squared = np.minimum(minimum_squared, squared)
        if step + 1 < count:
            scores = np.where(selected, -1.0, minimum_squared)
            current = int(np.argmax(scores))
    return output


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def build_patch_center_plan(
    *,
    cache_root: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    center_count: int = 64,
    workers: int = 1,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Build and atomically publish one FPS-prefix row per shape and view."""

    cache = Path(cache_root).resolve()
    count = _positive_integer(center_count, name="center_count")
    worker_count = _positive_integer(workers, name="workers")
    if worker_count > 256:
        raise ValueError("workers must not exceed 256")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")
    (
        sample_ids,
        point_count,
        cache_sha256,
        cache_format_version,
        view_count,
        sample_records,
        source_views_sha256,
    ) = _surface_cache_contract_any(cache)
    if count > point_count:
        raise ValueError("center_count must not exceed cache num_points")
    source_sha256 = _source_sha256()
    stage = None
    try:
        stage = _create_owned_stage(output_dir)

        def build_row(
            item: tuple[str, tuple[tuple[str, int, str], ...]],
        ) -> np.ndarray:
            _sample_id_value, records = item
            view_rows = []
            for record in records:
                relative_path, expected_bytes, expected_sha256 = record
                clean = _load_clean(
                    cache / relative_path,
                    point_count=point_count,
                    expected_file_bytes=expected_bytes,
                    expected_sha256=expected_sha256,
                )
                view_rows.append(_farthest_point_indices(clean, count))
            return np.ascontiguousarray(
                np.stack(view_rows),
                dtype=np.int32,
            )

        rows: list[np.ndarray] = []
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            iterator = executor.map(
                build_row,
                zip(sample_ids, sample_records),
            )
            for completed, (sample_id, row) in enumerate(
                zip(sample_ids, iterator),
                start=1,
            ):
                rows.append(row)
                if progress is not None:
                    progress(completed, len(sample_ids), sample_id)
        stacked = np.ascontiguousarray(np.stack(rows), dtype=np.int32)
        indices = (
            stacked[:, 0, :]
            if cache_format_version == 1
            else stacked
        )
        centers_path = stage.path / "centers.npy"
        with centers_path.open("xb") as stream:
            np.save(stream, indices, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        centers_bytes, centers_sha256 = _file_sha256(centers_path)
        if _source_sha256() != source_sha256:
            raise RuntimeError("patch-center source changed during build")
        if _surface_cache_contract_any(cache) != (
            sample_ids,
            point_count,
            cache_sha256,
            cache_format_version,
            view_count,
            sample_records,
            source_views_sha256,
        ):
            raise RuntimeError("surface-cache manifest changed during build")
        manifest: dict[str, object] = {
            "format": (
                PATCH_CENTER_PLAN_FORMAT
                if cache_format_version == 1
                else PATCH_CENTER_PLAN_V2_FORMAT
            ),
            "format_version": (
                PATCH_CENTER_PLAN_VERSION
                if cache_format_version == 1
                else PATCH_CENTER_PLAN_V2_VERSION
            ),
            "mode": PATCH_CENTER_MODE,
            "algorithm": _ALGORITHM,
            "shape_count": len(sample_ids),
            "sample_ids": list(sample_ids),
            "sample_ids_sha256": _sample_ids_digest(sample_ids),
            "point_count": point_count,
            "center_count": count,
            "start_index": 0,
            "tie_break": "lowest_point_index",
            "array_contract": {
                "container": "npy",
                "dtype": "int32",
                "shape": (
                    [len(sample_ids), count]
                    if cache_format_version == 1
                    else [len(sample_ids), view_count, count]
                ),
            },
            "source_cache_content_sha256": cache_sha256,
            "centers_sha256": centers_sha256,
            "centers_file_bytes": centers_bytes,
            "source_sha256": source_sha256,
        }
        if cache_format_version == 2:
            manifest["view_count"] = view_count
            manifest["source_views_sha256"] = source_views_sha256
        manifest["content_sha256"] = _manifest_digest(manifest)
        _write_json(stage.path / "manifest.json", manifest)
        _publish_owned_stage(stage)
        stage = None
        return manifest
    except BaseException:
        if stage is not None:
            _cleanup_owned_stage(stage)
        raise


def verify_patch_center_source_cache(
    cache_root: os.PathLike[str] | str,
    *,
    expected_content_sha256: str,
    expected_source_views_sha256: str | None = None,
    workers: int = 16,
) -> dict[str, object]:
    """Re-hash every planned clean file immediately before training.

    A plan proves what bytes were used while it was built.  This preflight
    closes the later gap where a ``clean.npy`` could change while the manifest
    and plan remained untouched.  It intentionally does not require meshes or
    split files, so it is much cheaper than rebuilding the formal cache.
    """

    cache = Path(cache_root).resolve()
    worker_count = _positive_integer(workers, name="workers")
    if worker_count > 256:
        raise ValueError("workers must not exceed 256")
    contract = _surface_cache_contract_any(cache)
    (
        sample_ids,
        _point_count,
        content_sha256,
        _cache_format_version,
        view_count,
        nested_records,
        source_views_sha256,
    ) = contract
    records = tuple(
        record
        for sample_records in nested_records
        for record in sample_records
    )
    if content_sha256 != _sha256(
        expected_content_sha256,
        name="expected source cache content_sha256",
    ):
        raise ValueError("source cache content SHA does not match plan")
    if (
        expected_source_views_sha256 is not None
        and source_views_sha256
        != _sha256(
            expected_source_views_sha256,
            name="expected source views SHA",
        )
    ):
        raise ValueError("source views SHA does not match the cache contract")

    def verify_file(record: tuple[str, int, str]) -> int:
        relative_path, expected_bytes, expected_sha256 = record
        path = cache / relative_path
        if path.is_symlink():
            raise ValueError(f"clean cache file must not be a symlink: {path}")
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ValueError(f"cannot read clean cache file: {path}") from error
        if (
            len(payload) != expected_bytes
            or hashlib.sha256(payload).hexdigest() != expected_sha256
        ):
            raise ValueError(
                f"clean cache file size/SHA does not match manifest: {path}"
            )
        return len(payload)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        sizes = tuple(executor.map(verify_file, records))
    if _surface_cache_contract_any(cache) != contract:
        raise RuntimeError("surface-cache manifest changed during verification")
    report = {
        "source_cache_content_sha256": content_sha256,
        "verified_file_count": len(records),
        "verified_file_bytes": sum(sizes),
    }
    if _cache_format_version == 2:
        report.update(
            {
                "source_views_sha256": source_views_sha256,
                "verified_shape_count": len(sample_ids),
                "verified_view_count": view_count,
            }
        )
    return report


def verify_patch_center_plan(
    plan_dir: os.PathLike[str] | str,
    *,
    expected_content_sha256: str | None = None,
    source_cache_content_sha256: str | None = None,
    expected_sample_ids: Sequence[str] | None = None,
    expected_point_count: int | None = None,
    expected_view_count: int | None = None,
) -> PatchCenterPlan:
    """Verify provenance, inventory, bytes, values, and row ordering."""

    root = Path(plan_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    inventory = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if inventory != {"manifest.json", "centers.npy"}:
        raise ValueError("patch-center plan file inventory is invalid")
    manifest = _read_json(
        root / "manifest.json",
        label="patch-center plan manifest",
    )
    is_v1 = (
        manifest.get("format") == PATCH_CENTER_PLAN_FORMAT
        and manifest.get("format_version") == PATCH_CENTER_PLAN_VERSION
    )
    is_v2 = (
        manifest.get("format") == PATCH_CENTER_PLAN_V2_FORMAT
        and manifest.get("format_version") == PATCH_CENTER_PLAN_V2_VERSION
    )
    expected_keys = _V2_MANIFEST_KEYS if is_v2 else _MANIFEST_KEYS
    if (
        (not is_v1 and not is_v2)
        or set(manifest) != expected_keys
        or manifest.get("mode") != PATCH_CENTER_MODE
        or manifest.get("algorithm") != _ALGORITHM
        or manifest.get("start_index") != 0
        or manifest.get("tie_break") != "lowest_point_index"
    ):
        raise ValueError("patch-center plan manifest fields are invalid")
    content_sha256 = _sha256(
        manifest.get("content_sha256"),
        name="patch-center content_sha256",
    )
    if _manifest_digest(manifest) != content_sha256:
        raise ValueError("patch-center plan content hash does not match")
    _sha256(manifest.get("source_sha256"), name="patch-center source SHA")
    if expected_content_sha256 is not None and content_sha256 != _sha256(
        expected_content_sha256,
        name="expected patch-center content_sha256",
    ):
        raise ValueError("patch-center plan content SHA does not match expected")
    cache_sha256 = _sha256(
        manifest.get("source_cache_content_sha256"),
        name="source cache content_sha256",
    )
    if source_cache_content_sha256 is not None and cache_sha256 != _sha256(
        source_cache_content_sha256,
        name="expected source cache content_sha256",
    ):
        raise ValueError("patch-center plan source cache does not match")
    sample_ids = _sample_ids(manifest.get("sample_ids"))
    shape_count = _positive_integer(
        manifest.get("shape_count"),
        name="patch-center shape_count",
    )
    point_count = _positive_integer(
        manifest.get("point_count"),
        name="patch-center point_count",
    )
    center_count = _positive_integer(
        manifest.get("center_count"),
        name="patch-center center_count",
    )
    view_count = (
        _positive_integer(
            manifest.get("view_count"),
            name="patch-center view_count",
        )
        if is_v2
        else 1
    )
    if view_count not in (1, 2):
        raise ValueError("patch-center view_count must be 1 or 2")
    if expected_view_count is not None and view_count != _positive_integer(
        expected_view_count,
        name="expected_view_count",
    ):
        raise ValueError("patch-center view_count does not match expected")
    source_views_sha256 = (
        _sha256(
            manifest.get("source_views_sha256"),
            name="patch-center source views SHA",
        )
        if is_v2
        else None
    )
    if shape_count != len(sample_ids):
        raise ValueError("patch-center shape_count does not match sample IDs")
    if manifest.get("sample_ids_sha256") != _sample_ids_digest(sample_ids):
        raise ValueError("patch-center sample ID hash does not match")
    if expected_sample_ids is not None:
        expected = tuple(_sample_id(value) for value in expected_sample_ids)
        if sample_ids != expected:
            raise ValueError("patch-center sample rows do not match expected")
    if expected_point_count is not None and point_count != _positive_integer(
        expected_point_count,
        name="expected_point_count",
    ):
        raise ValueError("patch-center point_count does not match expected")
    expected_shape = (
        [shape_count, view_count, center_count]
        if is_v2
        else [shape_count, center_count]
    )
    array_contract = manifest.get("array_contract")
    if array_contract != {
        "container": "npy",
        "dtype": "int32",
        "shape": expected_shape,
    }:
        raise ValueError("patch-center array contract is invalid")
    centers_path = root / "centers.npy"
    centers_bytes, centers_sha256 = _file_sha256(centers_path)
    if (
        centers_bytes
        != _positive_integer(
            manifest.get("centers_file_bytes"),
            name="centers_file_bytes",
        )
        or centers_sha256
        != _sha256(manifest.get("centers_sha256"), name="centers SHA")
    ):
        raise ValueError("patch-center centers SHA/size does not match")
    try:
        indices = np.load(
            centers_path,
            allow_pickle=False,
            mmap_mode="r",
        )
    except (OSError, ValueError) as error:
        raise ValueError("invalid patch-center centers.npy") from error
    if indices.dtype != np.int32 or list(indices.shape) != expected_shape:
        raise ValueError("patch-center centers.npy contract does not match")
    rows = indices.reshape((-1, center_count))
    if (
        int(indices.min()) < 0
        or int(indices.max()) >= point_count
        or not np.all(rows[:, 0] == 0)
        or any(len(set(row.tolist())) != center_count for row in rows)
    ):
        raise ValueError("patch-center indices are out of range or repeated")
    indices.flags.writeable = False
    return PatchCenterPlan(
        indices=indices,
        sample_ids=sample_ids,
        center_count=center_count,
        point_count=point_count,
        content_sha256=content_sha256,
        source_cache_content_sha256=cache_sha256,
        root=root,
        view_count=view_count,
        source_views_sha256=source_views_sha256,
    )


__all__ = [
    "PATCH_CENTER_MODE",
    "PATCH_CENTER_PLAN_FORMAT",
    "PATCH_CENTER_PLAN_VERSION",
    "PATCH_CENTER_PLAN_V2_FORMAT",
    "PATCH_CENTER_PLAN_V2_VERSION",
    "PatchCenterPlan",
    "build_patch_center_plan",
    "verify_patch_center_source_cache",
    "verify_patch_center_plan",
]

"""Publish a provenance-bound PGD1-output/clean cache for PGD2 training.

The published payload contains hard links only: ``pgd1_denoised.npy`` is the
PGD2 input and ``clean.npy`` is its same-index target.  The noisy cloud is not
duplicated because PGD2 never reads it; its digest is retained in the manifest
and is checked against the PGD1 inference input digest during publication.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import re
import stat
from numbers import Integral
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from .archive import (
    MODEL_RE,
    SYNSET_RE,
    _cleanup_owned_stage,
    _create_owned_stage,
)
from .surface_cache import (
    _durably_publish_owned_stage,
    _write_json,
    verify_surface_cache,
)


PGD2_PAIRED_CACHE_FORMAT = "pcdenoise_pgd2_paired_cache_v1"
PGD2_PAIRED_CACHE_VERSION = 1
PGD2_NOISY_CACHE_FORMAT = "pcdenoise_pgd2_noisy_cache_v1"
PGD1_INFERENCE_FORMAT = "pcdenoise_prediction_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PAIRED_KEYS = frozenset(
    (
        "format",
        "format_version",
        "status",
        "shape_count",
        "sample_count",
        "sample_ids",
        "point_count",
        "array_contract",
        "layout",
        "sources",
        "pgd1",
        "samples_sha256",
        "samples",
        "content_sha256",
    )
)
_SAMPLE_KEYS = frozenset(
    (
        "sample_id",
        "point_count",
        "pgd2_input",
        "clean_target",
        "noisy_source",
    )
)
_PUBLISHED_FILE_KEYS = frozenset(
    ("relative_path", "sha256", "file_bytes", "source_relative_path")
)
_NOISY_REFERENCE_KEYS = frozenset(
    ("relative_path", "sha256", "file_bytes")
)
_NOISY_MANIFEST_KEYS = frozenset(
    (
        "format",
        "format_version",
        "status",
        "epoch",
        "split_sha256",
        "shape_count",
        "shape_ids",
        "num_points",
        "array_contract",
        "noise_profile",
        "noise_policy",
        "base_seed",
        "seed_derivation",
        "rng",
        "source_clean_cache",
        "include_split",
        "samples_sha256",
        "samples",
        "content_sha256",
    )
)
_NOISY_SAMPLE_KEYS = frozenset(
    (
        "shape_id",
        "derived_seed",
        "noise_scale",
        "relative_path",
        "noisy_file_bytes",
        "noisy_sha256",
        "source_clean_sha256",
    )
)
_NOISY_CLEAN_SOURCE_KEYS = frozenset(
    (
        "format",
        "format_version",
        "content_sha256",
        "shape_count",
        "num_points",
    )
)
_NOISY_INCLUDE_SPLIT_KEYS = frozenset(
    (
        "file_sha256",
        "declared_split_sha256",
        "sample_ids_sha256",
        "sample_count",
    )
)
_INFERENCE_MANIFEST_KEYS = frozenset(
    (
        "format",
        "format_version",
        "status",
        "sample_count",
        "sample_ids",
        "patch_size",
        "seed_k",
        "patch_batch_size",
        "niters",
        "normalization_mode",
        "robust_quantile",
        "fusion_mode",
        "iteration_damping",
        "model_reference",
        "elapsed_seconds",
        "samples",
    )
)
_INFERENCE_SAMPLE_KEYS = frozenset(
    (
        "sample_id",
        "point_count",
        "input_sha256",
        "output_sha256",
        "relative_path",
        "elapsed_seconds",
        "details",
    )
)


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _sha256_value(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _sample_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("sample ID must be a string")
    parts = value.split("/")
    if (
        len(parts) != 2
        or SYNSET_RE.fullmatch(parts[0]) is None
        or MODEL_RE.fullmatch(parts[1]) is None
    ):
        raise ValueError(
            "sample ID must have form <8-digit synset>/<28-32 hex model>"
        )
    return value


def _sample_ids(value: object, *, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    result = [_sample_id(item) for item in value]
    if result != sorted(result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must be sorted and unique")
    return result


def _strict_relative_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError(f"{name} must be a normalized relative POSIX path")
    return value


def _no_duplicate_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _snapshot_regular(path: Path, *, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} is missing: {path}") from error
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    with path.open("rb") as stream:
        payload = stream.read()
    after = path.stat(follow_symlinks=False)
    if (
        after.st_dev != metadata.st_dev
        or after.st_ino != metadata.st_ino
        or after.st_size != metadata.st_size
        or after.st_mtime_ns != metadata.st_mtime_ns
    ):
        raise RuntimeError(f"{label} changed while reading: {path}")
    return payload


def _load_json(path: Path, *, label: str) -> tuple[dict[str, object], str]:
    payload = _snapshot_regular(path, label=label)
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_no_duplicate_object,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    return document, hashlib.sha256(payload).hexdigest()


def _require_directory(path: Path, *, label: str) -> Path:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} is missing: {path}") from error
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{label} must be a non-symlink directory: {path}")
    return path.resolve()


def _load_array(
    path: Path,
    *,
    expected_point_count: int,
    label: str,
) -> tuple[str, int]:
    payload = _snapshot_regular(path, label=label)
    try:
        values = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} must be a valid NPY file: {path}") from error
    if values.dtype != np.float32:
        raise ValueError(f"{label} must use float32: {path}")
    if values.shape != (expected_point_count, 3):
        raise ValueError(
            f"{label} must have shape ({expected_point_count},3): {path}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must contain only finite values: {path}")
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _manifest_digest(document: Mapping[str, object]) -> str:
    unsigned = dict(document)
    unsigned.pop("content_sha256", None)
    return _canonical_digest(unsigned)


def _verify_declared_digest(document: Mapping[str, object], *, label: str) -> str:
    claimed = _sha256_value(
        document.get("content_sha256"),
        name=f"{label} content_sha256",
    )
    if claimed != _manifest_digest(document):
        raise ValueError(f"{label} content_sha256 mismatch")
    return claimed


def _expected_source_path(sample_id: str, filename: str) -> str:
    return (PurePosixPath("shapenet") / sample_id / filename).as_posix()


def _regular_file_inventory(root: Path, *, label: str) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} contains a symlink: {path}")
        if stat.S_ISREG(metadata.st_mode):
            files.add(path.relative_to(root).as_posix())
        elif not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} contains an unsafe entry: {path}")
    return files


def _validate_noisy_cache(
    cache_dir: Path,
    *,
    clean_manifest: Mapping[str, object],
    clean_records: Mapping[str, Mapping[str, object]],
    expected_point_count: int,
) -> tuple[dict[str, object], str, dict[str, dict[str, object]]]:
    root = _require_directory(cache_dir, label="noisy cache")
    manifest, manifest_sha256 = _load_json(
        root / "manifest.json",
        label="noisy cache manifest",
    )
    if set(manifest) != _NOISY_MANIFEST_KEYS:
        raise ValueError("noisy cache manifest fields are invalid")
    if (
        manifest.get("format") != PGD2_NOISY_CACHE_FORMAT
        or manifest.get("format_version") != 1
    ):
        raise ValueError("noisy cache manifest format is invalid")
    if manifest.get("status") != "completed":
        raise ValueError("noisy cache status must be completed")
    _verify_declared_digest(manifest, label="noisy cache")
    ids = _sample_ids(manifest.get("shape_ids"), name="noisy shape_ids")
    if (
        _positive_integer(
            manifest.get("shape_count"),
            name="noisy shape_count",
        )
        != len(ids)
    ):
        raise ValueError("noisy cache shape_count is invalid")
    if manifest.get("num_points") != expected_point_count:
        raise ValueError("noisy cache num_points does not match")
    if manifest.get("array_contract") != {
        "container": "npy",
        "dtype": "float32",
        "shape": [expected_point_count, 3],
        "finite": True,
        "index_order": "preserved_from_clean",
    }:
        raise ValueError("noisy cache array contract is invalid")
    source_clean = manifest.get("source_clean_cache")
    if (
        not isinstance(source_clean, Mapping)
        or set(source_clean) != _NOISY_CLEAN_SOURCE_KEYS
    ):
        raise ValueError("noisy source_clean_cache is invalid")
    expected_clean_source = {
        "format": clean_manifest["format"],
        "format_version": clean_manifest["format_version"],
        "content_sha256": clean_manifest["content_sha256"],
        "shape_count": clean_manifest["shape_count"],
        "num_points": expected_point_count,
    }
    if dict(source_clean) != expected_clean_source:
        raise ValueError("noisy cache is not bound to the clean cache")
    include_split = manifest.get("include_split")
    if (
        not isinstance(include_split, Mapping)
        or set(include_split) != _NOISY_INCLUDE_SPLIT_KEYS
    ):
        raise ValueError("noisy include_split is invalid")
    _sha256_value(
        include_split.get("file_sha256"),
        name="noisy include-split file SHA256",
    )
    declared_split = include_split.get("declared_split_sha256")
    if declared_split is not None:
        declared_split = _sha256_value(
            declared_split,
            name="noisy declared split SHA256",
        )
    sample_ids_sha256 = _sha256_value(
        include_split.get("sample_ids_sha256"),
        name="noisy sample IDs SHA256",
    )
    if sample_ids_sha256 != _canonical_digest(ids):
        raise ValueError("noisy include-split sample IDs SHA256 mismatch")
    if include_split.get("sample_count") != len(ids):
        raise ValueError("noisy include-split sample count mismatch")
    split_sha256 = _sha256_value(
        manifest.get("split_sha256"),
        name="noisy split SHA256",
    )
    if split_sha256 != (declared_split or sample_ids_sha256):
        raise ValueError("noisy split SHA256 binding mismatch")
    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) != len(ids):
        raise ValueError("noisy cache samples/count is invalid")
    if manifest.get("samples_sha256") != _canonical_digest(raw_samples):
        raise ValueError("noisy cache samples_sha256 mismatch")

    records: dict[str, dict[str, object]] = {}
    expected_files = {"manifest.json"}
    for expected_id, raw_record in zip(ids, raw_samples):
        if (
            not isinstance(raw_record, Mapping)
            or set(raw_record) != _NOISY_SAMPLE_KEYS
        ):
            raise ValueError("noisy cache sample fields are invalid")
        shape_id = _sample_id(raw_record.get("shape_id"))
        if shape_id != expected_id:
            raise ValueError("noisy cache sample ordering is invalid")
        relative = _strict_relative_path(
            raw_record.get("relative_path"),
            name="noisy relative_path",
        )
        if relative != _expected_source_path(shape_id, "noisy.npy"):
            raise ValueError("noisy cache sample layout is invalid")
        clean_record = clean_records.get(shape_id)
        if clean_record is None:
            raise ValueError("noisy/clean ID universe mismatch")
        declared_clean_sha = _sha256_value(
            raw_record.get("source_clean_sha256"),
            name="noisy source clean SHA256",
        )
        if declared_clean_sha != clean_record["clean_sha256"]:
            raise ValueError("noisy source clean SHA256 mismatch")
        path = root / relative
        expected_files.add(relative)
        noisy_sha256, file_bytes = _load_array(
            path,
            expected_point_count=expected_point_count,
            label=f"noisy array {shape_id}",
        )
        if raw_record.get("noisy_file_bytes") != file_bytes:
            raise ValueError("noisy cache file byte count mismatch")
        if _sha256_value(
            raw_record.get("noisy_sha256"),
            name="noisy cache SHA256",
        ) != noisy_sha256:
            raise ValueError("noisy cache SHA256 mismatch")
        records[shape_id] = {
            "path": path,
            "relative_path": relative,
            "sha256": noisy_sha256,
            "file_bytes": file_bytes,
        }
    if _regular_file_inventory(root, label="noisy cache") != expected_files:
        raise ValueError("noisy cache file inventory is invalid")
    return manifest, manifest_sha256, records


def _validate_pgd1_output(
    output_dir: Path,
    *,
    noisy_records: Mapping[str, Mapping[str, object]],
    expected_ids: Sequence[str],
    expected_point_count: int,
    expected_checkpoint_sha256: str,
    expected_config_sha256: str,
) -> tuple[dict[str, object], str, dict[str, dict[str, object]]]:
    root = _require_directory(output_dir, label="PGD1 output")
    manifest, manifest_sha256 = _load_json(
        root / "inference_manifest.json",
        label="PGD1 inference manifest",
    )
    if set(manifest) != _INFERENCE_MANIFEST_KEYS:
        raise ValueError("PGD1 inference manifest fields are invalid")
    if (
        manifest.get("format") != PGD1_INFERENCE_FORMAT
        or manifest.get("format_version") != 1
    ):
        raise ValueError("PGD1 inference manifest format is invalid")
    if manifest.get("status") != "completed":
        raise ValueError("PGD1 inference status must be completed")
    ids = _sample_ids(manifest.get("sample_ids"), name="PGD1 sample_ids")
    if ids != list(expected_ids):
        raise ValueError("PGD1/noisy ID universe mismatch")
    if manifest.get("sample_count") != len(ids):
        raise ValueError("PGD1 inference sample_count is invalid")
    reference = manifest.get("model_reference")
    if not isinstance(reference, Mapping):
        raise ValueError("PGD1 model_reference is invalid")
    checkpoint_sha256 = _sha256_value(
        reference.get("checkpoint_sha256"),
        name="PGD1 checkpoint SHA256",
    )
    config_sha256 = _sha256_value(
        reference.get("config_sha256"),
        name="PGD1 config SHA256",
    )
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("PGD1 checkpoint SHA256 mismatch")
    if config_sha256 != expected_config_sha256:
        raise ValueError("PGD1 config SHA256 mismatch")
    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) != len(ids):
        raise ValueError("PGD1 inference samples/count is invalid")

    records: dict[str, dict[str, object]] = {}
    expected_files = {"inference_manifest.json"}
    for expected_id, raw_record in zip(ids, raw_samples):
        if (
            not isinstance(raw_record, Mapping)
            or set(raw_record) != _INFERENCE_SAMPLE_KEYS
        ):
            raise ValueError("PGD1 inference sample fields are invalid")
        sample_id = _sample_id(raw_record.get("sample_id"))
        if sample_id != expected_id:
            raise ValueError("PGD1 inference sample ordering is invalid")
        if raw_record.get("point_count") != expected_point_count:
            raise ValueError("PGD1 inference point_count is invalid")
        input_sha256 = _sha256_value(
            raw_record.get("input_sha256"),
            name="PGD1 input SHA256",
        )
        if input_sha256 != noisy_records[sample_id]["sha256"]:
            raise ValueError(f"PGD1 input SHA256 mismatch for {sample_id}")
        relative = _strict_relative_path(
            raw_record.get("relative_path"),
            name="PGD1 output relative_path",
        )
        if relative != _expected_source_path(sample_id, "denoised.npy"):
            raise ValueError("PGD1 inference output layout is invalid")
        path = root / relative
        expected_files.add(relative)
        output_sha256, file_bytes = _load_array(
            path,
            expected_point_count=expected_point_count,
            label=f"PGD1 output array {sample_id}",
        )
        declared_output = _sha256_value(
            raw_record.get("output_sha256"),
            name="PGD1 output SHA256",
        )
        if declared_output != output_sha256:
            raise ValueError(f"PGD1 output SHA256 mismatch for {sample_id}")
        records[sample_id] = {
            "path": path,
            "relative_path": relative,
            "sha256": output_sha256,
            "file_bytes": file_bytes,
        }
    if _regular_file_inventory(root, label="PGD1 output") != expected_files:
        raise ValueError("PGD1 output file inventory is invalid")
    return manifest, manifest_sha256, records


def _source_summary(
    manifest: Mapping[str, object],
    *,
    manifest_sha256: str,
    include_content: bool,
) -> dict[str, object]:
    result: dict[str, object] = {
        "format": manifest["format"],
        "format_version": manifest["format_version"],
        "manifest_sha256": manifest_sha256,
    }
    if include_content:
        result["content_sha256"] = manifest["content_sha256"]
    if "epoch" in manifest:
        result["epoch"] = manifest["epoch"]
    if "split_sha256" in manifest:
        result["split_sha256"] = manifest["split_sha256"]
    return result


def _hardlink(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError as error:
        if error.errno == errno.EXDEV:
            raise RuntimeError(
                "paired-cache hard links require clean cache, PGD1 output, "
                "and output_dir to be on the same filesystem"
            ) from error
        raise
    source_stat = source.stat(follow_symlinks=False)
    destination_stat = destination.stat(follow_symlinks=False)
    if (
        source_stat.st_dev != destination_stat.st_dev
        or source_stat.st_ino != destination_stat.st_ino
    ):
        raise RuntimeError("paired-cache hard link identity mismatch")


def build_pgd2_paired_cache(
    *,
    clean_cache: os.PathLike[str] | str,
    noisy_cache: os.PathLike[str] | str,
    pgd1_output: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    expected_point_count: int = 50_000,
    expected_pgd1_checkpoint_sha256: str,
    expected_pgd1_config_sha256: str,
) -> dict[str, object]:
    """Validate three aligned sources and atomically publish hardlinked pairs."""

    point_count = _positive_integer(
        expected_point_count,
        name="expected_point_count",
    )
    expected_checkpoint = _sha256_value(
        expected_pgd1_checkpoint_sha256,
        name="expected PGD1 checkpoint SHA256",
    )
    expected_config = _sha256_value(
        expected_pgd1_config_sha256,
        name="expected PGD1 config SHA256",
    )
    clean_root = _require_directory(Path(clean_cache), label="clean cache")
    clean_manifest = verify_surface_cache(clean_root, verify_files=True)
    clean_manifest_sha256 = hashlib.sha256(
        _snapshot_regular(
            clean_root / "manifest.json",
            label="clean cache manifest",
        )
    ).hexdigest()
    if clean_manifest.get("num_points") != point_count:
        raise ValueError("clean cache num_points does not match")
    clean_ids = _sample_ids(
        clean_manifest.get("shape_ids"),
        name="clean shape_ids",
    )
    if clean_manifest.get("shape_count") != len(clean_ids):
        raise ValueError("clean cache shape_count is invalid")
    clean_records = {
        record["shape_id"]: record for record in clean_manifest["samples"]
    }
    noisy_manifest, noisy_manifest_sha256, noisy_records = (
        _validate_noisy_cache(
            Path(noisy_cache),
            clean_manifest=clean_manifest,
            clean_records=clean_records,
            expected_point_count=point_count,
        )
    )
    noisy_ids = list(noisy_records)
    pgd1_manifest, pgd1_manifest_sha256, pgd1_records = (
        _validate_pgd1_output(
            Path(pgd1_output),
            noisy_records=noisy_records,
            expected_ids=noisy_ids,
            expected_point_count=point_count,
            expected_checkpoint_sha256=expected_checkpoint,
            expected_config_sha256=expected_config,
        )
    )

    sample_records: list[dict[str, object]] = []
    link_plan: list[tuple[Path, str]] = []
    for sample_id in noisy_ids:
        clean = clean_records[sample_id]
        clean_source_relative = _strict_relative_path(
            clean["relative_path"],
            name="clean relative_path",
        )
        if clean_source_relative != _expected_source_path(sample_id, "clean.npy"):
            raise ValueError("clean cache sample layout is invalid")
        clean_path = clean_root / clean_source_relative
        pgd1 = pgd1_records[sample_id]
        noisy = noisy_records[sample_id]
        input_relative = _expected_source_path(sample_id, "pgd1_denoised.npy")
        target_relative = _expected_source_path(sample_id, "clean.npy")
        record = {
            "sample_id": sample_id,
            "point_count": point_count,
            "pgd2_input": {
                "relative_path": input_relative,
                "sha256": pgd1["sha256"],
                "file_bytes": pgd1["file_bytes"],
                "source_relative_path": pgd1["relative_path"],
            },
            "clean_target": {
                "relative_path": target_relative,
                "sha256": clean["clean_sha256"],
                "file_bytes": clean["clean_file_bytes"],
                "source_relative_path": clean_source_relative,
            },
            "noisy_source": {
                "relative_path": noisy["relative_path"],
                "sha256": noisy["sha256"],
                "file_bytes": noisy["file_bytes"],
            },
        }
        sample_records.append(record)
        link_plan.extend(
            (
                (pgd1["path"], input_relative),
                (clean_path, target_relative),
            )
        )

    reference = pgd1_manifest["model_reference"]
    pgd1_binding: dict[str, object] = {
        "checkpoint_sha256": expected_checkpoint,
        "config_sha256": expected_config,
        "inference_manifest_sha256": pgd1_manifest_sha256,
    }
    if "checkpoint_step" in reference:
        pgd1_binding["checkpoint_step"] = reference["checkpoint_step"]
    manifest: dict[str, object] = {
        "format": PGD2_PAIRED_CACHE_FORMAT,
        "format_version": PGD2_PAIRED_CACHE_VERSION,
        "status": "completed",
        "shape_count": len(noisy_ids),
        "sample_count": len(noisy_ids),
        "sample_ids": noisy_ids,
        "point_count": point_count,
        "array_contract": {
            "container": "npy",
            "dtype": "float32",
            "shape": [point_count, 3],
            "finite": True,
            "index_order": "preserved_from_clean_through_pgd1",
        },
        "layout": {
            "sample_directory": "shapenet/<synset>/<model>",
            "pgd2_input_filename": "pgd1_denoised.npy",
            "clean_target_filename": "clean.npy",
            "storage_mode": "hardlink",
            "noisy_payload": "manifest_reference_only",
        },
        "sources": {
            "clean_cache": _source_summary(
                clean_manifest,
                manifest_sha256=clean_manifest_sha256,
                include_content=True,
            ),
            "noisy_cache": _source_summary(
                noisy_manifest,
                manifest_sha256=noisy_manifest_sha256,
                include_content=True,
            ),
            "pgd1_inference": _source_summary(
                pgd1_manifest,
                manifest_sha256=pgd1_manifest_sha256,
                include_content=False,
            ),
        },
        "pgd1": pgd1_binding,
        "samples_sha256": _canonical_digest(sample_records),
        "samples": sample_records,
    }
    manifest["content_sha256"] = _manifest_digest(manifest)

    stage = _create_owned_stage(output_dir)
    try:
        for source, relative in link_plan:
            destination = stage.path / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            _hardlink(source, destination)
        _write_json(stage.path / "manifest.json", manifest)
        _durably_publish_owned_stage(stage)
    except BaseException:
        _cleanup_owned_stage(stage)
        raise
    return verify_pgd2_paired_cache(
        output_dir,
        expected_content_sha256=manifest["content_sha256"],
        expected_pgd1_checkpoint_sha256=expected_checkpoint,
        expected_pgd1_config_sha256=expected_config,
        verify_files=True,
    )


def _validated_published_file(
    value: object,
    *,
    name: str,
    expected_relative_path: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _PUBLISHED_FILE_KEYS:
        raise ValueError(f"{name} record is invalid")
    relative = _strict_relative_path(value["relative_path"], name=f"{name} path")
    if relative != expected_relative_path:
        raise ValueError(f"{name} layout is invalid")
    _strict_relative_path(value["source_relative_path"], name=f"{name} source")
    _sha256_value(value["sha256"], name=f"{name} SHA256")
    _positive_integer(value["file_bytes"], name=f"{name} file_bytes")
    return value


def verify_pgd2_paired_cache(
    cache_dir: os.PathLike[str] | str,
    *,
    expected_content_sha256: str | None = None,
    expected_pgd1_checkpoint_sha256: str | None = None,
    expected_pgd1_config_sha256: str | None = None,
    verify_files: bool = True,
) -> dict[str, object]:
    """Verify a paired manifest and, optionally, every linked NPY payload."""

    root = _require_directory(Path(cache_dir), label="PGD2 paired cache")
    manifest, _ = _load_json(
        root / "manifest.json",
        label="PGD2 paired manifest",
    )
    if set(manifest) != _PAIRED_KEYS:
        raise ValueError("PGD2 paired manifest fields are invalid")
    if (
        manifest.get("format") != PGD2_PAIRED_CACHE_FORMAT
        or manifest.get("format_version") != PGD2_PAIRED_CACHE_VERSION
        or manifest.get("status") != "completed"
    ):
        raise ValueError("PGD2 paired manifest format/status is invalid")
    content_sha256 = _verify_declared_digest(
        manifest,
        label="PGD2 paired cache",
    )
    if expected_content_sha256 is not None and content_sha256 != _sha256_value(
        expected_content_sha256,
        name="expected paired content SHA256",
    ):
        raise ValueError("PGD2 paired content SHA256 mismatch")
    point_count = _positive_integer(
        manifest.get("point_count"),
        name="paired point_count",
    )
    ids = _sample_ids(manifest.get("sample_ids"), name="paired sample_ids")
    if (
        manifest.get("shape_count") != len(ids)
        or manifest.get("sample_count") != len(ids)
    ):
        raise ValueError("PGD2 paired sample count is invalid")
    expected_contract = {
        "container": "npy",
        "dtype": "float32",
        "shape": [point_count, 3],
        "finite": True,
        "index_order": "preserved_from_clean_through_pgd1",
    }
    if manifest.get("array_contract") != expected_contract:
        raise ValueError("PGD2 paired array contract is invalid")
    expected_layout = {
        "sample_directory": "shapenet/<synset>/<model>",
        "pgd2_input_filename": "pgd1_denoised.npy",
        "clean_target_filename": "clean.npy",
        "storage_mode": "hardlink",
        "noisy_payload": "manifest_reference_only",
    }
    if manifest.get("layout") != expected_layout:
        raise ValueError("PGD2 paired layout is invalid")
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or set(sources) != {
        "clean_cache",
        "noisy_cache",
        "pgd1_inference",
    }:
        raise ValueError("PGD2 paired sources are invalid")
    clean_source = sources["clean_cache"]
    noisy_source = sources["noisy_cache"]
    inference_source = sources["pgd1_inference"]
    if (
        not isinstance(clean_source, dict)
        or set(clean_source)
        != {
            "format",
            "format_version",
            "manifest_sha256",
            "content_sha256",
            "split_sha256",
        }
        or clean_source.get("format")
        not in {"pcdenoise_train_surface_cache_v1", "pcdenoise_train_surface_cache_v2"}
    ):
        raise ValueError("paired clean-cache source is invalid")
    if (
        not isinstance(noisy_source, dict)
        or set(noisy_source)
        != {
            "format",
            "format_version",
            "manifest_sha256",
            "content_sha256",
            "epoch",
            "split_sha256",
        }
        or noisy_source.get("format") != PGD2_NOISY_CACHE_FORMAT
        or noisy_source.get("format_version") != 1
    ):
        raise ValueError("paired noisy-cache source is invalid")
    if (
        not isinstance(inference_source, dict)
        or set(inference_source)
        != {"format", "format_version", "manifest_sha256"}
        or inference_source.get("format") != PGD1_INFERENCE_FORMAT
        or inference_source.get("format_version") != 1
    ):
        raise ValueError("paired PGD1-inference source is invalid")
    for label, source in (
        ("clean-cache", clean_source),
        ("noisy-cache", noisy_source),
        ("PGD1-inference", inference_source),
    ):
        _sha256_value(
            source["manifest_sha256"],
            name=f"paired {label} manifest SHA256",
        )
    _sha256_value(
        clean_source["content_sha256"],
        name="paired clean-cache content SHA256",
    )
    _sha256_value(
        clean_source["split_sha256"],
        name="paired clean-cache split SHA256",
    )
    _sha256_value(
        noisy_source["content_sha256"],
        name="paired noisy-cache content SHA256",
    )
    _sha256_value(
        noisy_source["split_sha256"],
        name="paired noisy-cache split SHA256",
    )
    _positive_integer(noisy_source["epoch"], name="paired noisy-cache epoch")
    pgd1 = manifest.get("pgd1")
    expected_pgd1_keys = {
        "checkpoint_sha256",
        "config_sha256",
        "inference_manifest_sha256",
    }
    if not isinstance(pgd1, dict) or set(pgd1) not in (
        expected_pgd1_keys,
        expected_pgd1_keys | {"checkpoint_step"},
    ):
        raise ValueError("PGD2 paired PGD1 binding is invalid")
    checkpoint_sha256 = _sha256_value(
        pgd1.get("checkpoint_sha256"),
        name="paired PGD1 checkpoint SHA256",
    )
    config_sha256 = _sha256_value(
        pgd1.get("config_sha256"),
        name="paired PGD1 config SHA256",
    )
    _sha256_value(
        pgd1.get("inference_manifest_sha256"),
        name="paired PGD1 inference manifest SHA256",
    )
    if (
        pgd1["inference_manifest_sha256"]
        != inference_source["manifest_sha256"]
    ):
        raise ValueError("paired PGD1 inference manifest binding mismatch")
    if "checkpoint_step" in pgd1:
        _positive_integer(pgd1["checkpoint_step"], name="PGD1 checkpoint_step")
    if (
        expected_pgd1_checkpoint_sha256 is not None
        and checkpoint_sha256
        != _sha256_value(
            expected_pgd1_checkpoint_sha256,
            name="expected PGD1 checkpoint SHA256",
        )
    ):
        raise ValueError("PGD1 checkpoint SHA256 mismatch")
    if (
        expected_pgd1_config_sha256 is not None
        and config_sha256
        != _sha256_value(
            expected_pgd1_config_sha256,
            name="expected PGD1 config SHA256",
        )
    ):
        raise ValueError("PGD1 config SHA256 mismatch")
    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) != len(ids):
        raise ValueError("PGD2 paired samples/count is invalid")
    if manifest.get("samples_sha256") != _canonical_digest(raw_samples):
        raise ValueError("PGD2 paired samples_sha256 mismatch")

    expected_files = {"manifest.json"}
    for expected_id, raw_record in zip(ids, raw_samples):
        if not isinstance(raw_record, dict) or set(raw_record) != _SAMPLE_KEYS:
            raise ValueError("PGD2 paired sample record is invalid")
        if raw_record.get("sample_id") != expected_id:
            raise ValueError("PGD2 paired sample ordering is invalid")
        if raw_record.get("point_count") != point_count:
            raise ValueError("PGD2 paired sample point_count is invalid")
        pgd2_input = _validated_published_file(
            raw_record["pgd2_input"],
            name="PGD2 input",
            expected_relative_path=_expected_source_path(
                expected_id,
                "pgd1_denoised.npy",
            ),
        )
        clean_target = _validated_published_file(
            raw_record["clean_target"],
            name="clean target",
            expected_relative_path=_expected_source_path(expected_id, "clean.npy"),
        )
        if pgd2_input["source_relative_path"] != _expected_source_path(
            expected_id,
            "denoised.npy",
        ):
            raise ValueError("PGD2 input source layout is invalid")
        if clean_target["source_relative_path"] != _expected_source_path(
            expected_id,
            "clean.npy",
        ):
            raise ValueError("clean target source layout is invalid")
        noisy = raw_record["noisy_source"]
        if not isinstance(noisy, dict) or set(noisy) != _NOISY_REFERENCE_KEYS:
            raise ValueError("noisy source record is invalid")
        noisy_relative = _strict_relative_path(
            noisy["relative_path"],
            name="noisy source path",
        )
        if noisy_relative != _expected_source_path(expected_id, "noisy.npy"):
            raise ValueError("noisy source layout is invalid")
        _sha256_value(noisy["sha256"], name="noisy source SHA256")
        _positive_integer(noisy["file_bytes"], name="noisy source file_bytes")
        for label, file_record in (
            ("PGD2 input", pgd2_input),
            ("clean target", clean_target),
        ):
            relative = file_record["relative_path"]
            expected_files.add(relative)
            if verify_files:
                actual_sha256, file_bytes = _load_array(
                    root / relative,
                    expected_point_count=point_count,
                    label=f"{label} {expected_id}",
                )
                if actual_sha256 != file_record["sha256"]:
                    raise ValueError(f"{label} SHA256 mismatch for {expected_id}")
                if file_bytes != file_record["file_bytes"]:
                    raise ValueError(f"{label} file byte count mismatch")
    if verify_files:
        actual_files = _regular_file_inventory(root, label="PGD2 paired cache")
        if actual_files != expected_files:
            raise ValueError("PGD2 paired cache file inventory is invalid")
    return manifest


__all__ = [
    "PGD2_PAIRED_CACHE_FORMAT",
    "PGD2_PAIRED_CACHE_VERSION",
    "build_pgd2_paired_cache",
    "verify_pgd2_paired_cache",
]

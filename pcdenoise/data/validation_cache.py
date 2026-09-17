"""Build an immutable, deterministic clean/noisy validation cache."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from .archive import (
    MODEL_RE,
    SYNSET_RE,
    _cleanup_owned_stage,
    _create_owned_stage,
    _publish_owned_stage,
    sha256_file,
)
from .mesh_dataset import fit_unit_sphere, load_obj_bytes, sample_surface
from .noise import NOISE_PROFILES, add_isotropic_gaussian, add_noise


VALIDATION_SPLIT_FORMAT = "pcdenoise_shape_ids_v1"
VALIDATION_CACHE_FORMAT = "pcdenoise_validation_cache_v2"
VALIDATION_CACHE_VERSION = 2
VALIDATION_CACHE_V3_FORMAT = "pcdenoise_validation_cache_v3"
VALIDATION_CACHE_V3_VERSION = 3
_GENERATION_CONTRACT_KEYS = (
    "format",
    "format_version",
    "shape_count",
    "shape_ids",
    "num_points",
    "array_contract",
    "noise_profile",
    "noise_sigma",
    "seed",
    "seed_derivation",
    "rng",
    "split_sha256",
    "validation_split_sha256",
    "split_manifest_sha256",
    "source_sha256",
    "samples_sha256",
    "samples",
)
_GENERATION_CONTRACT_V3_KEYS = (
    "format",
    "format_version",
    "shape_count",
    "shape_ids",
    "num_points",
    "array_contract",
    "noise_profile",
    "noise_policy",
    "seed",
    "seed_derivation",
    "rng",
    "split_sha256",
    "validation_split_sha256",
    "split_manifest_sha256",
    "source_sha256",
    "samples_sha256",
    "samples",
)
_MANIFEST_KEYS = frozenset(
    (*_GENERATION_CONTRACT_KEYS, "content_sha256")
)
_MANIFEST_V3_KEYS = frozenset(
    (*_GENERATION_CONTRACT_V3_KEYS, "content_sha256")
)
_SAMPLE_KEYS = frozenset(
    (
        "shape_id",
        "derived_seed",
        "mesh_sha256",
        "clean_sha256",
        "noisy_sha256",
        "normalization_center",
        "normalization_scale",
    )
)
_SAMPLE_V3_KEYS = frozenset(
    (
        "shape_id",
        "derived_seed",
        "mesh_sha256",
        "clean_sha256",
        "noisy_sha256",
        "normalization_center",
        "normalization_scale",
        "noise_profile",
        "noise_scale",
        "scale_bin",
    )
)
_SEED_DERIVATION = (
    "le_u64(first8(sha256(le_u64(global_seed)||NUL||"
    "ascii(shape_id))))"
)
_RNG_KEYS = frozenset(
    (
        "library",
        "numpy_version",
        "generator",
        "bit_generator",
        "stream_policy",
    )
)
_RNG_STREAM_POLICY = (
    "one_per_shape_surface_sampling_then_gaussian_noise"
)
_RNG_V3_STREAM_POLICY = (
    "one_per_shape_surface_sampling_then_named_noise_at_assigned_scale"
)
_NOISE_POLICY_KEYS = frozenset(
    (
        "assignment",
        "bin_edges",
        "noise_max",
        "noise_min",
        "scale_bins",
        "scale_rule",
    )
)


def _positive_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _seed(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or not 0 <= int(value) < 2**64
    ):
        raise ValueError("seed must be an integer in [0, 2**64)")
    return int(value)


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return converted


def _v3_noise_policy(
    *,
    noise_profile: object,
    scale: object,
    noise_min: object,
    noise_max: object,
    scale_bins: object,
    shape_count: int,
) -> dict[str, object]:
    if noise_profile not in NOISE_PROFILES:
        raise ValueError(
            f"noise_profile must be one of {NOISE_PROFILES}"
        )
    range_values = (noise_min, noise_max, scale_bins)
    if scale is not None:
        if any(value is not None for value in range_values):
            raise ValueError(
                "scale conflicts with noise_min/noise_max/scale_bins"
            )
        fixed_scale = _positive_finite(scale, name="scale")
        return {
            "assignment": "fixed",
            "bin_edges": [fixed_scale, fixed_scale],
            "noise_max": fixed_scale,
            "noise_min": fixed_scale,
            "scale_bins": 1,
            "scale_rule": "fixed",
        }
    if any(value is None for value in range_values):
        raise ValueError(
            "stratified noise requires noise_min, noise_max, and scale_bins"
        )
    low = _positive_finite(noise_min, name="noise_min")
    high = _positive_finite(noise_max, name="noise_max")
    if high <= low:
        raise ValueError("noise_max must be greater than noise_min")
    bins = _positive_integer(scale_bins, name="scale_bins")
    if bins > shape_count:
        raise ValueError("scale_bins cannot exceed shape_count")
    width = (high - low) / bins
    edges = [
        low + width * index
        for index in range(bins)
    ] + [high]
    return {
        "assignment": "stratified",
        "bin_edges": edges,
        "noise_max": high,
        "noise_min": low,
        "scale_bins": bins,
        "scale_rule": "midpoint_of_sorted_shape_index_mod_bin",
    }


def _assigned_noise_scale(
    policy: dict[str, object],
    *,
    shape_index: int,
) -> tuple[float, int]:
    if policy["assignment"] == "fixed":
        return float(policy["noise_min"]), 0
    bins = int(policy["scale_bins"])
    scale_bin = shape_index % bins
    edges = policy["bin_edges"]
    if not isinstance(edges, list):
        raise RuntimeError("noise policy bin edges are invalid")
    return (
        (float(edges[scale_bin]) + float(edges[scale_bin + 1])) * 0.5,
        scale_bin,
    )


def _shape_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("every shape ID must be a string")
    parts = value.split("/")
    if (
        len(parts) != 2
        or SYNSET_RE.fullmatch(parts[0]) is None
        or MODEL_RE.fullmatch(parts[1]) is None
    ):
        raise ValueError(
            "shape ID must have form <8-digit synset>/<28-32 hex model>"
        )
    return value


def _sha256_value(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_validation_split(
    path: Path,
    *,
    expected_file_sha256: str | None,
) -> tuple[list[str], str, str, str | None]:
    payload, validation_file_sha256 = _snapshot_regular_file(path)
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid validation split JSON: {path}") from error
    if not isinstance(document, dict):
        raise ValueError("validation split must be a JSON object")
    if (
        document.get("format") != VALIDATION_SPLIT_FORMAT
        or document.get("format_version") != 1
        or document.get("split") != "validation"
    ):
        raise ValueError("validation split metadata is invalid")
    raw_ids = document.get("shape_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("validation split shape_ids must be a nonempty list")
    shape_ids = [_shape_id(value) for value in raw_ids]
    if len(set(shape_ids)) != len(shape_ids):
        raise ValueError("validation split contains duplicate shape IDs")
    if (
        isinstance(document.get("count"), bool)
        or document.get("count") != len(shape_ids)
    ):
        raise ValueError("validation split count does not match shape_ids")
    split_sha256 = _sha256_value(
        document.get("split_sha256"),
        name="validation split_sha256",
    )

    if expected_file_sha256 is not None:
        expected = _sha256_value(
            expected_file_sha256,
            name="expected validation split SHA",
        )
        if validation_file_sha256 != expected:
            raise ValueError(
                "validation split actual SHA does not match expected SHA"
            )

    sibling_path = path.with_name("manifest.json")
    sibling_sha256: str | None = None
    if os.path.lexists(sibling_path):
        sibling_payload, sibling_sha256 = _snapshot_regular_file(
            sibling_path
        )
        try:
            sibling = json.loads(sibling_payload.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid sibling split manifest JSON: {sibling_path}"
            ) from error
        if not isinstance(sibling, dict) or sibling.get("format_version") != 1:
            raise ValueError("sibling split manifest metadata is invalid")
        sibling_claim = _sha256_value(
            sibling.get("split_sha256"),
            name="sibling split_sha256",
        )
        unsigned_sibling = dict(sibling)
        unsigned_sibling.pop("split_sha256", None)
        if _canonical_digest(unsigned_sibling) != sibling_claim:
            raise ValueError("sibling split manifest SHA is invalid")
        sibling_validation = sibling.get("validation")
        sibling_counts = sibling.get("counts")
        if (
            not isinstance(sibling_validation, list)
            or sibling_validation != shape_ids
            or not isinstance(sibling_counts, dict)
            or sibling_counts.get("validation") != len(shape_ids)
        ):
            raise ValueError(
                "sibling split manifest validation IDs/count do not match"
            )
        if sibling_claim != split_sha256:
            raise ValueError(
                "sibling split manifest SHA does not match validation split"
            )
    elif expected_file_sha256 is None:
        raise ValueError(
            "standalone validation split requires an expected file SHA"
        )
    return (
        sorted(shape_ids),
        split_sha256,
        validation_file_sha256,
        sibling_sha256,
    )


def _derived_seed(global_seed: int, shape_id: str) -> int:
    digest = hashlib.sha256(
        global_seed.to_bytes(8, "little", signed=False)
        + b"\0"
        + shape_id.encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _save_npy(path: Path, array: np.ndarray) -> None:
    values = np.asarray(array)
    if values.dtype != np.float32 or values.ndim != 2 or values.shape[1] != 3:
        raise RuntimeError("validation array contract was violated")
    if not np.isfinite(values).all():
        raise RuntimeError("validation array contains non-finite values")
    with path.open("xb") as stream:
        np.save(stream, np.ascontiguousarray(values), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _snapshot_regular_file(path: Path) -> tuple[bytes, str]:
    """Read one no-follow regular-file snapshot and hash those exact bytes."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"source must be a regular file: {path}")
        digest = hashlib.sha256()
        chunks = []
        while True:
            chunk = os.read(descriptor, 8 << 20)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
        ):
            raise RuntimeError(f"source changed while reading: {path}")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise RuntimeError(f"source size changed while reading: {path}")
        return payload, digest.hexdigest()
    finally:
        os.close(descriptor)


def _load_mesh_snapshot(
    mesh_path: Path,
) -> tuple[object, str]:
    """Parse and hash one immutable byte snapshot, never reopening its source."""

    payload, mesh_sha256 = _snapshot_regular_file(mesh_path)
    mesh = load_obj_bytes(payload)
    return mesh, mesh_sha256


def _write_json(path: Path, value: Any) -> None:
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _content_digest(records: list[dict[str, object]]) -> str:
    canonical = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _generation_source_hashes() -> dict[str, str]:
    source_paths = {
        "pcdenoise/data/mesh_dataset.py": Path(__file__).with_name(
            "mesh_dataset.py"
        ),
        "pcdenoise/data/noise.py": Path(__file__).with_name("noise.py"),
        "pcdenoise/data/validation_cache.py": Path(__file__),
    }
    hashes = {}
    for name, path in source_paths.items():
        _, digest = _snapshot_regular_file(path)
        hashes[name] = digest
    return dict(sorted(hashes.items()))


def _generation_contract_digest(manifest: dict[str, object]) -> str:
    """Hash the path/time-independent contract needed to reproduce a cache."""

    contract_keys = (
        _GENERATION_CONTRACT_V3_KEYS
        if manifest.get("format") == VALIDATION_CACHE_V3_FORMAT
        else _GENERATION_CONTRACT_KEYS
    )
    try:
        payload = {
            key: manifest[key] for key in contract_keys
        }
    except KeyError as error:
        raise ValueError(
            f"generation contract is missing field {error.args[0]!r}"
        ) from error
    return _canonical_digest(payload)


def _load_npy_snapshot(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
    num_points: int,
) -> np.ndarray:
    payload, actual_sha256 = _snapshot_regular_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA does not match manifest: {path}")
    try:
        values = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid {label} NPY: {path}") from error
    if values.shape != (num_points, 3) or values.dtype != np.float32:
        raise ValueError(f"{label} array contract does not match manifest")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")
    return np.ascontiguousarray(values)


def _verified_v3_noise_policy(
    raw_policy: object,
    *,
    noise_profile: object,
    shape_count: int,
) -> dict[str, object]:
    if (
        not isinstance(raw_policy, dict)
        or set(raw_policy) != _NOISE_POLICY_KEYS
    ):
        raise ValueError("validation cache noise policy fields are invalid")
    assignment = raw_policy.get("assignment")
    if assignment == "fixed":
        expected = _v3_noise_policy(
            noise_profile=noise_profile,
            scale=raw_policy.get("noise_min"),
            noise_min=None,
            noise_max=None,
            scale_bins=None,
            shape_count=shape_count,
        )
    elif assignment == "stratified":
        expected = _v3_noise_policy(
            noise_profile=noise_profile,
            scale=None,
            noise_min=raw_policy.get("noise_min"),
            noise_max=raw_policy.get("noise_max"),
            scale_bins=raw_policy.get("scale_bins"),
            shape_count=shape_count,
        )
    else:
        raise ValueError("validation cache noise assignment is invalid")
    if raw_policy != expected:
        raise ValueError("validation cache noise policy is invalid")
    return expected


def _verify_validation_cache_v3(
    root: Path,
    manifest: dict[str, object],
    *,
    mesh_root: os.PathLike[str] | str | None,
    validation_split: os.PathLike[str] | str | None,
    expected_content_sha256: str | None,
    verify_files: bool,
) -> dict[str, object]:
    if set(manifest) != _MANIFEST_V3_KEYS:
        raise ValueError("validation cache manifest fields are invalid")
    shape_count = _positive_integer(
        manifest.get("shape_count"),
        name="manifest shape_count",
    )
    count = _positive_integer(
        manifest.get("num_points"),
        name="manifest num_points",
    )
    noise_profile = manifest.get("noise_profile")
    policy = _verified_v3_noise_policy(
        manifest.get("noise_policy"),
        noise_profile=noise_profile,
        shape_count=shape_count,
    )
    if manifest.get("seed_derivation") != _SEED_DERIVATION:
        raise ValueError("validation cache generation metadata is invalid")
    global_seed = _seed(manifest.get("seed"))
    _sha256_value(manifest.get("split_sha256"), name="manifest split SHA")
    _sha256_value(
        manifest.get("validation_split_sha256"),
        name="manifest validation split SHA",
    )
    split_manifest_sha256 = manifest.get("split_manifest_sha256")
    if split_manifest_sha256 is not None:
        _sha256_value(
            split_manifest_sha256,
            name="manifest split manifest SHA",
        )
    claimed_content = _sha256_value(
        manifest.get("content_sha256"),
        name="validation cache content SHA",
    )
    if expected_content_sha256 is not None:
        expected = _sha256_value(
            expected_content_sha256,
            name="expected validation cache content SHA",
        )
        if claimed_content != expected:
            raise ValueError(
                "validation cache content SHA does not match expected SHA"
            )
    if _generation_contract_digest(manifest) != claimed_content:
        raise ValueError("validation cache content SHA is invalid")

    shape_ids_raw = manifest.get("shape_ids")
    records = manifest.get("samples")
    if not isinstance(shape_ids_raw, list) or not isinstance(records, list):
        raise ValueError("validation cache shape/sample lists are invalid")
    shape_ids = [_shape_id(value) for value in shape_ids_raw]
    if (
        shape_ids != sorted(shape_ids)
        or len(set(shape_ids)) != len(shape_ids)
        or shape_count != len(shape_ids)
        or len(records) != len(shape_ids)
    ):
        raise ValueError("validation cache shape ordering/count is invalid")
    if manifest.get("samples_sha256") != _content_digest(records):
        raise ValueError("validation cache samples SHA is invalid")

    array_contract = manifest.get("array_contract")
    if array_contract != {
        "container": "npy",
        "coordinate_frame": (
            "clean_bbox_center_max_radius_unit_sphere"
        ),
        "dtype": "float32",
        "shape": [count, 3],
    }:
        raise ValueError("validation cache array contract is invalid")
    rng = manifest.get("rng")
    if (
        not isinstance(rng, dict)
        or set(rng) != _RNG_KEYS
        or rng.get("library") != "numpy"
        or rng.get("generator") != "numpy.random.Generator"
        or not isinstance(rng.get("numpy_version"), str)
        or not rng.get("numpy_version")
        or not isinstance(rng.get("bit_generator"), str)
        or not rng.get("bit_generator")
        or rng.get("stream_policy") != _RNG_V3_STREAM_POLICY
    ):
        raise ValueError("validation cache RNG contract is invalid")
    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != {
        "pcdenoise/data/mesh_dataset.py",
        "pcdenoise/data/noise.py",
        "pcdenoise/data/validation_cache.py",
    }:
        raise ValueError("validation cache source hashes are invalid")
    for name, value in source_hashes.items():
        _sha256_value(value, name=f"source hash {name}")

    if validation_split is not None:
        (
            split_shape_ids,
            split_sha256,
            validation_split_sha256,
            actual_split_manifest_sha256,
        ) = _read_validation_split(
            Path(validation_split),
            expected_file_sha256=manifest.get(
                "validation_split_sha256"
            ),
        )
        if (
            split_shape_ids != shape_ids
            or split_sha256 != manifest.get("split_sha256")
            or validation_split_sha256
            != manifest.get("validation_split_sha256")
            or actual_split_manifest_sha256
            != manifest.get("split_manifest_sha256")
        ):
            raise ValueError(
                "validation cache split binding does not match source"
            )

    mesh_base = Path(mesh_root) if mesh_root is not None else None
    expected_files = {"manifest.json"}
    for index, (shape_id, raw_record) in enumerate(
        zip(shape_ids, records)
    ):
        if (
            not isinstance(raw_record, dict)
            or set(raw_record) != _SAMPLE_V3_KEYS
        ):
            raise ValueError("validation cache sample record is invalid")
        if raw_record.get("shape_id") != shape_id:
            raise ValueError(
                f"validation cache sample order differs at index {index}"
            )
        derived_seed = _seed(raw_record.get("derived_seed"))
        if derived_seed != _derived_seed(global_seed, shape_id):
            raise ValueError(f"derived seed is invalid for {shape_id}")
        if raw_record.get("noise_profile") != noise_profile:
            raise ValueError(
                f"noise profile is invalid for {shape_id}"
            )
        expected_scale, expected_bin = _assigned_noise_scale(
            policy,
            shape_index=index,
        )
        actual_scale = _positive_finite(
            raw_record.get("noise_scale"),
            name=f"noise scale for {shape_id}",
        )
        raw_bin = raw_record.get("scale_bin")
        if (
            isinstance(raw_bin, bool)
            or not isinstance(raw_bin, Integral)
            or int(raw_bin) != expected_bin
            or actual_scale != expected_scale
        ):
            raise ValueError(
                f"noise scale/bin assignment is invalid for {shape_id}"
            )
        mesh_sha256 = _sha256_value(
            raw_record.get("mesh_sha256"),
            name=f"mesh SHA for {shape_id}",
        )
        clean_sha256 = _sha256_value(
            raw_record.get("clean_sha256"),
            name=f"clean SHA for {shape_id}",
        )
        noisy_sha256 = _sha256_value(
            raw_record.get("noisy_sha256"),
            name=f"noisy SHA for {shape_id}",
        )
        raw_center = raw_record.get("normalization_center")
        if (
            not isinstance(raw_center, list)
            or len(raw_center) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, Real)
                for value in raw_center
            )
        ):
            raise ValueError(
                f"normalization contract is invalid for {shape_id}"
            )
        center = np.asarray(raw_center, dtype=np.float64)
        _positive_finite(
            raw_record.get("normalization_scale"),
            name=f"normalization scale for {shape_id}",
        )
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError(
                f"normalization contract is invalid for {shape_id}"
            )

        relative_clean = f"shapenet/{shape_id}/clean.npy"
        relative_noisy = f"shapenet/{shape_id}/noisy.npy"
        expected_files.update((relative_clean, relative_noisy))
        clean = None
        if verify_files:
            clean = _load_npy_snapshot(
                root / relative_clean,
                expected_sha256=clean_sha256,
                label="clean",
                num_points=count,
            )
            _load_npy_snapshot(
                root / relative_noisy,
                expected_sha256=noisy_sha256,
                label="noisy",
                num_points=count,
            )
        if clean is not None:
            normalized_transform = fit_unit_sphere(clean)
            if (
                not np.allclose(
                    normalized_transform.center,
                    np.zeros(3),
                    rtol=0.0,
                    atol=2.0e-6,
                )
                or not math.isclose(
                    normalized_transform.scale,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=2.0e-6,
                )
            ):
                raise ValueError(
                    f"clean array is not in the unit-sphere frame for "
                    f"{shape_id}"
                )
        if mesh_base is not None:
            mesh_path = (
                mesh_base
                / shape_id
                / "models"
                / "model_normalized.obj"
            )
            _, actual_mesh_sha256 = _snapshot_regular_file(mesh_path)
            if actual_mesh_sha256 != mesh_sha256:
                raise ValueError(
                    f"mesh SHA does not match manifest for {shape_id}"
                )

    if verify_files:
        actual_files = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            raise ValueError("validation cache file inventory is invalid")
    return manifest


def verify_validation_cache(
    cache_dir: os.PathLike[str] | str,
    *,
    mesh_root: os.PathLike[str] | str | None = None,
    validation_split: os.PathLike[str] | str | None = None,
    expected_content_sha256: str | None = None,
    verify_files: bool = True,
) -> dict[str, object]:
    """Fail closed unless a published v2/v3 cache matches its contract."""

    if not isinstance(verify_files, bool):
        raise ValueError("verify_files must be a bool")
    root = Path(cache_dir)
    manifest_payload, _ = _snapshot_regular_file(root / "manifest.json")
    try:
        manifest = json.loads(manifest_payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid validation cache manifest JSON") from error
    if (
        isinstance(manifest, dict)
        and manifest.get("format") == VALIDATION_CACHE_V3_FORMAT
        and manifest.get("format_version") == VALIDATION_CACHE_V3_VERSION
    ):
        return _verify_validation_cache_v3(
            root,
            manifest,
            mesh_root=mesh_root,
            validation_split=validation_split,
            expected_content_sha256=expected_content_sha256,
            verify_files=verify_files,
        )
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != VALIDATION_CACHE_FORMAT
        or manifest.get("format_version") != VALIDATION_CACHE_VERSION
    ):
        raise ValueError("validation cache manifest metadata is invalid")
    if set(manifest) != _MANIFEST_KEYS:
        raise ValueError("validation cache manifest fields are invalid")
    shape_count = _positive_integer(
        manifest.get("shape_count"),
        name="manifest shape_count",
    )
    count = _positive_integer(
        manifest.get("num_points"),
        name="manifest num_points",
    )
    noise_sigma = _positive_finite(
        manifest.get("noise_sigma"),
        name="manifest noise_sigma",
    )
    if (
        manifest.get("noise_profile")
        != "isotropic_gaussian_unit_sphere"
        or manifest.get("seed_derivation") != _SEED_DERIVATION
    ):
        raise ValueError("validation cache generation metadata is invalid")
    global_seed = _seed(manifest.get("seed"))
    _sha256_value(manifest.get("split_sha256"), name="manifest split SHA")
    _sha256_value(
        manifest.get("validation_split_sha256"),
        name="manifest validation split SHA",
    )
    split_manifest_sha256 = manifest.get("split_manifest_sha256")
    if split_manifest_sha256 is not None:
        _sha256_value(
            split_manifest_sha256,
            name="manifest split manifest SHA",
        )
    claimed_content = _sha256_value(
        manifest.get("content_sha256"),
        name="validation cache content SHA",
    )
    if expected_content_sha256 is not None:
        expected = _sha256_value(
            expected_content_sha256,
            name="expected validation cache content SHA",
        )
        if claimed_content != expected:
            raise ValueError(
                "validation cache content SHA does not match expected SHA"
            )
    if _generation_contract_digest(manifest) != claimed_content:
        raise ValueError("validation cache content SHA is invalid")

    shape_ids_raw = manifest.get("shape_ids")
    records = manifest.get("samples")
    if not isinstance(shape_ids_raw, list) or not isinstance(records, list):
        raise ValueError("validation cache shape/sample lists are invalid")
    shape_ids = [_shape_id(value) for value in shape_ids_raw]
    if (
        shape_ids != sorted(shape_ids)
        or len(set(shape_ids)) != len(shape_ids)
        or shape_count != len(shape_ids)
        or len(records) != len(shape_ids)
    ):
        raise ValueError("validation cache shape ordering/count is invalid")
    if manifest.get("samples_sha256") != _content_digest(records):
        raise ValueError("validation cache samples SHA is invalid")

    array_contract = manifest.get("array_contract")
    if array_contract != {
        "container": "npy",
        "dtype": "float32",
        "shape": [count, 3],
    }:
        raise ValueError("validation cache array contract is invalid")
    rng = manifest.get("rng")
    if (
        not isinstance(rng, dict)
        or set(rng) != _RNG_KEYS
        or rng.get("library") != "numpy"
        or rng.get("generator") != "numpy.random.Generator"
        or not isinstance(rng.get("numpy_version"), str)
        or not rng.get("numpy_version")
        or not isinstance(rng.get("bit_generator"), str)
        or not rng.get("bit_generator")
        or rng.get("stream_policy") != _RNG_STREAM_POLICY
    ):
        raise ValueError("validation cache RNG contract is invalid")
    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != {
        "pcdenoise/data/mesh_dataset.py",
        "pcdenoise/data/noise.py",
        "pcdenoise/data/validation_cache.py",
    }:
        raise ValueError("validation cache source hashes are invalid")
    for name, value in source_hashes.items():
        _sha256_value(value, name=f"source hash {name}")

    if validation_split is not None:
        (
            split_shape_ids,
            split_sha256,
            validation_split_sha256,
            split_manifest_sha256,
        ) = _read_validation_split(
            Path(validation_split),
            expected_file_sha256=manifest.get(
                "validation_split_sha256"
            ),
        )
        if (
            split_shape_ids != shape_ids
            or split_sha256 != manifest.get("split_sha256")
            or validation_split_sha256
            != manifest.get("validation_split_sha256")
            or split_manifest_sha256
            != manifest.get("split_manifest_sha256")
        ):
            raise ValueError(
                "validation cache split binding does not match source"
            )

    mesh_base = Path(mesh_root) if mesh_root is not None else None
    expected_files = {"manifest.json"}
    for index, (shape_id, raw_record) in enumerate(
        zip(shape_ids, records)
    ):
        if (
            not isinstance(raw_record, dict)
            or set(raw_record) != _SAMPLE_KEYS
        ):
            raise ValueError("validation cache sample record is invalid")
        if raw_record.get("shape_id") != shape_id:
            raise ValueError(
                f"validation cache sample order differs at index {index}"
            )
        derived_seed = _seed(raw_record.get("derived_seed"))
        if derived_seed != _derived_seed(global_seed, shape_id):
            raise ValueError(f"derived seed is invalid for {shape_id}")
        mesh_sha256 = _sha256_value(
            raw_record.get("mesh_sha256"),
            name=f"mesh SHA for {shape_id}",
        )
        clean_sha256 = _sha256_value(
            raw_record.get("clean_sha256"),
            name=f"clean SHA for {shape_id}",
        )
        noisy_sha256 = _sha256_value(
            raw_record.get("noisy_sha256"),
            name=f"noisy SHA for {shape_id}",
        )
        raw_center = raw_record.get("normalization_center")
        if (
            not isinstance(raw_center, list)
            or len(raw_center) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, Real)
                for value in raw_center
            )
        ):
            raise ValueError(
                f"normalization contract is invalid for {shape_id}"
            )
        center = np.asarray(raw_center, dtype=np.float64)
        scale = _positive_finite(
            raw_record.get("normalization_scale"),
            name=f"normalization scale for {shape_id}",
        )
        if (
            center.shape != (3,)
            or not np.isfinite(center).all()
        ):
            raise ValueError(
                f"normalization contract is invalid for {shape_id}"
            )

        relative_clean = f"shapenet/{shape_id}/clean.npy"
        relative_noisy = f"shapenet/{shape_id}/noisy.npy"
        expected_files.update((relative_clean, relative_noisy))
        clean = None
        if verify_files:
            clean = _load_npy_snapshot(
                root / relative_clean,
                expected_sha256=clean_sha256,
                label="clean",
                num_points=count,
            )
            _load_npy_snapshot(
                root / relative_noisy,
                expected_sha256=noisy_sha256,
                label="noisy",
                num_points=count,
            )
        if clean is not None:
            transform = fit_unit_sphere(clean)
            if (
                not np.array_equal(transform.center, center)
                or transform.scale != scale
            ):
                raise ValueError(
                    f"normalization metadata differs for {shape_id}"
                )
        if mesh_base is not None:
            mesh_path = (
                mesh_base
                / shape_id
                / "models"
                / "model_normalized.obj"
            )
            _, actual_mesh_sha256 = _snapshot_regular_file(mesh_path)
            if actual_mesh_sha256 != mesh_sha256:
                raise ValueError(
                    f"mesh SHA does not match manifest for {shape_id}"
                )

    if verify_files:
        actual_files = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        if actual_files != expected_files:
            raise ValueError("validation cache file inventory is invalid")
    return manifest


def build_validation_cache(
    *,
    mesh_root: os.PathLike[str] | str,
    validation_split: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    num_points: int = 50_000,
    sigma: float | None = None,
    seed: int = 20260726,
    expected_validation_split_sha256: str | None = None,
    noise_profile: str | None = None,
    scale: float | None = None,
    noise_min: float | None = None,
    noise_max: float | None = None,
    scale_bins: int | None = None,
) -> dict[str, object]:
    """Generate one deterministic paired cloud per validation mesh.

    Calls without ``noise_profile`` retain the legacy v2 fixed-Gaussian
    contract and raw-frame arrays.  Explicit named-profile calls publish v3
    normalized arrays with either one fixed scale or deterministic stratified
    midpoint scales.  Each shape receives an order-independent derived seed.
    """

    count = _positive_integer(num_points, name="num_points")
    v3_requested = noise_profile is not None
    v3_arguments = (scale, noise_min, noise_max, scale_bins)
    if not v3_requested:
        if any(value is not None for value in v3_arguments):
            raise ValueError(
                "noise_profile is required for v3 noise options"
            )
        noise_sigma = _positive_finite(
            0.015 if sigma is None else sigma,
            name="sigma",
        )
    else:
        if sigma is not None:
            raise ValueError("sigma conflicts with named v3 noise options")
        noise_sigma = None
    global_seed = _seed(seed)
    root = Path(mesh_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    split_path = Path(validation_split)
    (
        shape_ids,
        split_sha256,
        validation_split_sha256,
        split_manifest_sha256,
    ) = _read_validation_split(
        split_path,
        expected_file_sha256=expected_validation_split_sha256,
    )
    policy = (
        _v3_noise_policy(
            noise_profile=noise_profile,
            scale=scale,
            noise_min=noise_min,
            noise_max=noise_max,
            scale_bins=scale_bins,
            shape_count=len(shape_ids),
        )
        if v3_requested
        else None
    )

    mesh_paths: dict[str, Path] = {}
    for shape_id in shape_ids:
        mesh_path = root / shape_id / "models" / "model_normalized.obj"
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        mesh_paths[shape_id] = mesh_path

    stage = _create_owned_stage(output_dir)
    try:
        records: list[dict[str, object]] = []
        for shape_index, shape_id in enumerate(shape_ids):
            per_shape_seed = _derived_seed(global_seed, shape_id)
            generator = np.random.default_rng(per_shape_seed)
            mesh_path = mesh_paths[shape_id]
            mesh, mesh_sha256 = _load_mesh_snapshot(mesh_path)
            clean = sample_surface(mesh, count, generator)
            transform = fit_unit_sphere(clean)
            normalized_clean = transform.apply(clean)
            if v3_requested:
                if policy is None or noise_profile is None:
                    raise RuntimeError("v3 noise policy was not initialized")
                assigned_scale, scale_bin = _assigned_noise_scale(
                    policy,
                    shape_index=shape_index,
                )
                normalized_noisy, actual_scale = add_noise(
                    normalized_clean,
                    generator,
                    profile=noise_profile,
                    scale=assigned_scale,
                )
                if actual_scale != assigned_scale:
                    raise RuntimeError(
                        "fixed validation noise scale changed"
                    )
                clean_to_save = normalized_clean
                noisy_to_save = normalized_noisy
            else:
                normalized_noisy, actual_sigma = (
                    add_isotropic_gaussian(
                        normalized_clean,
                        generator,
                        sigma=noise_sigma,
                    )
                )
                if actual_sigma != noise_sigma:
                    raise RuntimeError("fixed validation sigma changed")
                clean_to_save = clean
                noisy_to_save = transform.restore(normalized_noisy)

            sample_dir = stage.path / "shapenet" / shape_id
            sample_dir.mkdir(parents=True, exist_ok=False)
            clean_path = sample_dir / "clean.npy"
            noisy_path = sample_dir / "noisy.npy"
            _save_npy(clean_path, clean_to_save)
            _save_npy(noisy_path, noisy_to_save)
            record: dict[str, object] = {
                "shape_id": shape_id,
                "derived_seed": per_shape_seed,
                "mesh_sha256": mesh_sha256,
                "clean_sha256": sha256_file(clean_path),
                "noisy_sha256": sha256_file(noisy_path),
                "normalization_center": [
                    float(value) for value in transform.center
                ],
                "normalization_scale": float(transform.scale),
            }
            if v3_requested:
                record.update(
                    {
                        "noise_profile": noise_profile,
                        "noise_scale": assigned_scale,
                        "scale_bin": scale_bin,
                    }
                )
            records.append(record)

        if v3_requested:
            if policy is None or noise_profile is None:
                raise RuntimeError("v3 noise policy was not initialized")
            manifest: dict[str, object] = {
                "format": VALIDATION_CACHE_V3_FORMAT,
                "format_version": VALIDATION_CACHE_V3_VERSION,
                "shape_count": len(shape_ids),
                "shape_ids": shape_ids,
                "num_points": count,
                "array_contract": {
                    "container": "npy",
                    "coordinate_frame": (
                        "clean_bbox_center_max_radius_unit_sphere"
                    ),
                    "dtype": "float32",
                    "shape": [count, 3],
                },
                "noise_profile": noise_profile,
                "noise_policy": policy,
                "seed": global_seed,
                "seed_derivation": _SEED_DERIVATION,
                "rng": {
                    "library": "numpy",
                    "numpy_version": np.__version__,
                    "generator": "numpy.random.Generator",
                    "bit_generator": type(
                        np.random.default_rng(0).bit_generator
                    ).__name__,
                    "stream_policy": _RNG_V3_STREAM_POLICY,
                },
                "split_sha256": split_sha256,
                "validation_split_sha256": validation_split_sha256,
                "split_manifest_sha256": split_manifest_sha256,
                "source_sha256": _generation_source_hashes(),
                "samples_sha256": _content_digest(records),
                "samples": records,
            }
        else:
            manifest = {
                "format": VALIDATION_CACHE_FORMAT,
                "format_version": VALIDATION_CACHE_VERSION,
                "shape_count": len(shape_ids),
                "shape_ids": shape_ids,
                "num_points": count,
                "array_contract": {
                    "container": "npy",
                    "dtype": "float32",
                    "shape": [count, 3],
                },
                "noise_profile": "isotropic_gaussian_unit_sphere",
                "noise_sigma": noise_sigma,
                "seed": global_seed,
                "seed_derivation": _SEED_DERIVATION,
                "rng": {
                    "library": "numpy",
                    "numpy_version": np.__version__,
                    "generator": "numpy.random.Generator",
                    "bit_generator": type(
                        np.random.default_rng(0).bit_generator
                    ).__name__,
                    "stream_policy": _RNG_STREAM_POLICY,
                },
                "split_sha256": split_sha256,
                "validation_split_sha256": validation_split_sha256,
                "split_manifest_sha256": split_manifest_sha256,
                "source_sha256": _generation_source_hashes(),
                "samples_sha256": _content_digest(records),
                "samples": records,
            }
        manifest["content_sha256"] = _generation_contract_digest(manifest)
        _write_json(stage.path / "manifest.json", manifest)
        published = _publish_owned_stage(stage)
        if published != Path(output_dir).resolve():
            raise RuntimeError("validation cache published to unexpected path")
        return manifest
    except BaseException:
        _cleanup_owned_stage(stage)
        raise

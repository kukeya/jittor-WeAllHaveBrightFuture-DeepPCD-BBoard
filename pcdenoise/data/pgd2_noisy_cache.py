"""Deterministic, resumable whole-cloud Laplace inputs for frozen PGD1."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import io
import json
import math
import os
import re
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral, Real
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .noise import add_starter_laplace


PGD2_NOISY_CACHE_FORMAT = "pcdenoise_pgd2_noisy_cache_v1"
PGD2_NOISY_CACHE_VERSION = 1
_SEED_DOMAIN = b"pcdenoise:pgd2-noisy-cache-v1\0"
_SEED_DERIVATION = (
    "le_u128(first16(sha256(ascii(pcdenoise:pgd2-noisy-cache-v1)"
    "||NUL||le_u64(base_seed)||le_u64(epoch)||le_u64(id_length)"
    "||ascii(shape_id))))"
)
_SHAPE_ID_RE = re.compile(r"^[0-9]{8}/[0-9a-f]{28,32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100
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


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _uint64(value: object, *, name: str, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < int(positive)
        or int(value) >= 2**64
    ):
        qualifier = "positive " if positive else "nonnegative "
        raise ValueError(f"{name} must be a {qualifier}integer smaller than 2**64")
    return int(value)


def _scale(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _shape_id(value: object) -> str:
    if not isinstance(value, str) or _SHAPE_ID_RE.fullmatch(value) is None:
        raise ValueError(
            "shape ID must have form <8-digit synset>/<28-32 lowercase hex model>"
        )
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _regular_file_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"source must be a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 8 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _json_from_bytes(payload: bytes, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {name} JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def pgd2_noisy_seed(base_seed: int, *, epoch: int, shape_id: str) -> int:
    """Derive one order- and worker-independent NumPy seed per epoch/shape."""

    seed = _uint64(base_seed, name="base_seed")
    epoch_index = _uint64(epoch, name="epoch", positive=True)
    identifier = _shape_id(shape_id).encode("ascii")
    digest = hashlib.sha256(
        _SEED_DOMAIN
        + seed.to_bytes(8, "little", signed=False)
        + epoch_index.to_bytes(8, "little", signed=False)
        + len(identifier).to_bytes(8, "little", signed=False)
        + identifier
    ).digest()
    return int.from_bytes(digest[:16], "little", signed=False)


def _clean_manifest(root: Path) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    payload = _regular_file_bytes(root / "manifest.json")
    manifest = _json_from_bytes(payload, name="clean-cache manifest")
    claimed = _sha256(manifest.get("content_sha256"), name="clean content SHA")
    unsigned = dict(manifest)
    unsigned.pop("content_sha256", None)
    if _canonical_sha256(unsigned) != claimed:
        raise ValueError("clean-cache content SHA is invalid")
    raw_ids = manifest.get("shape_ids")
    raw_samples = manifest.get("samples")
    if not isinstance(raw_ids, list) or not isinstance(raw_samples, list):
        raise ValueError("clean-cache shape_ids/samples are invalid")
    shape_ids = [_shape_id(value) for value in raw_ids]
    if shape_ids != sorted(shape_ids) or len(shape_ids) != len(set(shape_ids)):
        raise ValueError("clean-cache shape_ids must be sorted and unique")
    if manifest.get("shape_count") != len(shape_ids) or len(raw_samples) != len(shape_ids):
        raise ValueError("clean-cache shape count is invalid")
    records: dict[str, dict[str, object]] = {}
    for expected_id, raw in zip(shape_ids, raw_samples):
        if not isinstance(raw, Mapping) or raw.get("shape_id") != expected_id:
            raise ValueError("clean-cache sample order is invalid")
        record = dict(raw)
        expected_path = f"shapenet/{expected_id}/clean.npy"
        if record.get("relative_path") != expected_path:
            raise ValueError(f"clean-cache relative path is invalid for {expected_id}")
        _sha256(record.get("clean_sha256"), name=f"clean SHA for {expected_id}")
        records[expected_id] = record
    return manifest, records


def _include_ids(path: Path) -> tuple[list[str], str, str | None]:
    payload = _regular_file_bytes(path)
    file_sha256 = hashlib.sha256(payload).hexdigest()
    declared_split_sha256 = None
    if path.suffix.lower() == ".json":
        document = _json_from_bytes(payload, name="include split")
        if (
            document.get("format") != "pcdenoise_shape_ids_v1"
            or document.get("format_version") != 1
            or document.get("split") != "train"
        ):
            raise ValueError("include split JSON metadata is invalid")
        raw_ids = document.get("shape_ids")
        if not isinstance(raw_ids, list):
            raise ValueError("include split JSON must contain shape_ids")
        if document.get("count") != len(raw_ids):
            raise ValueError("include split count does not match shape_ids")
        if "split_sha256" in document:
            declared_split_sha256 = _sha256(
                document["split_sha256"], name="declared split SHA"
            )
            unsigned = dict(document)
            unsigned.pop("split_sha256")
            if _canonical_sha256(unsigned) != declared_split_sha256:
                raise ValueError("include split declared split SHA is invalid")
    else:
        try:
            raw_ids = [line.strip() for line in payload.decode("ascii").splitlines()]
        except UnicodeError as error:
            raise ValueError("include split must be ASCII") from error
        if any(not value for value in raw_ids):
            raise ValueError("include split must not contain blank lines")
    ids = [_shape_id(value) for value in raw_ids]
    if not ids:
        raise ValueError("include split must not be empty")
    if len(ids) != len(set(ids)):
        raise ValueError("include split contains duplicate shape IDs")
    return sorted(ids), file_sha256, declared_split_sha256


def _write_json_exclusive(path: Path, value: object) -> None:
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ).encode("ascii") + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _append_record(path: Path, record: Mapping[str, object]) -> None:
    payload = _canonical_bytes(dict(record)) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise OSError("short write to PGD2 noisy-cache progress ledger")
            view = view[count:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_progress(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = _regular_file_bytes(path)
    complete_length = payload.rfind(b"\n") + 1
    complete = payload[:complete_length]
    records = []
    for line in complete.splitlines():
        try:
            value = json.loads(line.decode("ascii"))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("completed progress record is invalid") from error
        if not isinstance(value, dict):
            raise ValueError("completed progress record must be an object")
        records.append(value)
    if complete_length != len(payload):
        with path.open("r+b") as stream:
            stream.truncate(complete_length)
            stream.flush()
            os.fsync(stream.fileno())
    return records


def _write_noisy(path: Path, values: np.ndarray) -> tuple[int, str]:
    array = np.asarray(values)
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != 3:
        raise RuntimeError("PGD2 noisy array contract was violated")
    if not np.isfinite(array).all():
        raise RuntimeError("PGD2 noisy array contains non-finite values")
    payload_stream = io.BytesIO()
    np.save(
        payload_stream,
        np.ascontiguousarray(array),
        allow_pickle=False,
    )
    payload = payload_stream.getvalue()
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    file_bytes = path.stat().st_size
    if file_bytes != len(payload):
        raise RuntimeError("PGD2 noisy NPY byte length changed during publication")
    return file_bytes, payload_sha256


def _validate_noisy_file(
    path: Path,
    *,
    expected_point_count: int,
    expected_file_bytes: object,
    expected_sha256: object,
    label: str,
) -> None:
    file_bytes = _positive_integer(
        expected_file_bytes,
        name=f"{label} file bytes",
    )
    claimed_sha256 = _sha256(
        expected_sha256,
        name=f"{label} SHA256",
    )
    try:
        payload = _regular_file_bytes(path)
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} is not a regular readable file: {path}") from error
    if len(payload) != file_bytes:
        raise ValueError(f"{label} file byte count mismatch")
    if hashlib.sha256(payload).hexdigest() != claimed_sha256:
        raise ValueError(f"{label} SHA256 mismatch")
    try:
        values = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} is not a valid NPY array") from error
    if values.dtype != np.float32:
        raise ValueError(f"{label} dtype must be float32")
    if values.shape != (expected_point_count, 3):
        raise ValueError(
            f"{label} shape must be ({expected_point_count},3)"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must contain only finite values")


def _regular_file_inventory(root: Path, *, label: str) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} inventory contains a symlink: {path}")
        if stat.S_ISREG(metadata.st_mode):
            files.add(path.relative_to(root).as_posix())
        elif not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} inventory contains an unsafe entry: {path}")
    return files


def _rename_noreplace(source: Path, destination: Path) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        number = ctypes.get_errno()
        if number in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(destination)
        raise OSError(number, os.strerror(number), destination)


def _cleanup_resume_state(state: Path) -> None:
    if not state.exists():
        return
    for name in (".build_contract.json", ".progress.jsonl"):
        path = state / name
        if path.exists():
            path.unlink()
    if any(state.iterdir()):
        raise RuntimeError(f"refusing to remove unknown resume-state entries: {state}")
    state.rmdir()


def _existing_completed(
    path: Path,
    contract: Mapping[str, object],
    *,
    clean_records: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    manifest = _json_from_bytes(
        _regular_file_bytes(path / "manifest.json"), name="PGD2 noisy manifest"
    )
    claimed = _sha256(manifest.get("content_sha256"), name="noisy content SHA")
    unsigned = dict(manifest)
    unsigned.pop("content_sha256", None)
    if _canonical_sha256(unsigned) != claimed or manifest.get("status") != "completed":
        raise ValueError("existing PGD2 noisy cache is not completed and valid")
    if (
        set(manifest) != _NOISY_MANIFEST_KEYS
        or manifest.get("format") != PGD2_NOISY_CACHE_FORMAT
        or manifest.get("format_version") != PGD2_NOISY_CACHE_VERSION
    ):
        raise ValueError("existing PGD2 noisy cache manifest fields are invalid")
    for key in (
        "epoch",
        "base_seed",
        "shape_ids_sha256",
        "split_sha256",
        "source_clean_content_sha256",
        "include_split_file_sha256",
        "scale_min",
        "scale_max",
        "num_points",
    ):
        if contract[key] != {
            "epoch": manifest.get("epoch"),
            "base_seed": manifest.get("base_seed"),
            "shape_ids_sha256": manifest.get("include_split", {}).get(
                "sample_ids_sha256"
            ),
            "split_sha256": manifest.get("split_sha256"),
            "source_clean_content_sha256": manifest.get(
                "source_clean_cache", {}
            ).get("content_sha256"),
            "include_split_file_sha256": manifest.get("include_split", {}).get(
                "file_sha256"
            ),
            "scale_min": manifest.get("noise_policy", {}).get("scale_min"),
            "scale_max": manifest.get("noise_policy", {}).get("scale_max"),
            "num_points": manifest.get("num_points"),
        }[key]:
            raise ValueError("existing PGD2 noisy cache contract differs")

    raw_ids = manifest.get("shape_ids")
    raw_samples = manifest.get("samples")
    if not isinstance(raw_ids, list) or not isinstance(raw_samples, list):
        raise ValueError("existing PGD2 noisy cache IDs/samples are invalid")
    shape_ids = [_shape_id(value) for value in raw_ids]
    if (
        shape_ids != sorted(shape_ids)
        or len(shape_ids) != len(set(shape_ids))
        or len(shape_ids) != contract["shape_count"]
        or len(raw_samples) != len(shape_ids)
        or _canonical_sha256(shape_ids) != contract["shape_ids_sha256"]
    ):
        raise ValueError("existing PGD2 noisy cache ID universe is invalid")
    if manifest.get("samples_sha256") != _canonical_sha256(raw_samples):
        raise ValueError("existing PGD2 noisy cache samples SHA256 mismatch")
    point_count = int(contract["num_points"])
    if manifest.get("array_contract") != {
        "container": "npy",
        "dtype": "float32",
        "shape": [point_count, 3],
        "finite": True,
        "index_order": "preserved_from_clean",
    }:
        raise ValueError("existing PGD2 noisy cache array contract is invalid")

    expected_files = {"manifest.json"}
    for shape_id, raw_record in zip(shape_ids, raw_samples):
        if not isinstance(raw_record, Mapping) or set(raw_record) != _NOISY_SAMPLE_KEYS:
            raise ValueError("existing PGD2 noisy cache sample record is invalid")
        expected_relative = f"shapenet/{shape_id}/noisy.npy"
        expected_seed = pgd2_noisy_seed(
            int(contract["base_seed"]),
            epoch=int(contract["epoch"]),
            shape_id=shape_id,
        )
        expected_scale = float(
            np.random.default_rng(expected_seed).uniform(
                float(contract["scale_min"]),
                float(contract["scale_max"]),
            )
        )
        clean_record = clean_records.get(shape_id)
        if (
            raw_record.get("shape_id") != shape_id
            or raw_record.get("relative_path") != expected_relative
            or raw_record.get("derived_seed") != expected_seed
            or raw_record.get("noise_scale") != expected_scale
            or clean_record is None
            or raw_record.get("source_clean_sha256")
            != clean_record.get("clean_sha256")
        ):
            raise ValueError("existing PGD2 noisy cache sample binding is invalid")
        _validate_noisy_file(
            path / expected_relative,
            expected_point_count=point_count,
            expected_file_bytes=raw_record.get("noisy_file_bytes"),
            expected_sha256=raw_record.get("noisy_sha256"),
            label=f"existing noisy sample {shape_id}",
        )
        expected_files.add(expected_relative)
    if _regular_file_inventory(path, label="existing PGD2 noisy cache") != expected_files:
        raise ValueError("existing PGD2 noisy cache inventory mismatch")
    return manifest


def build_pgd2_noisy_cache(
    *,
    clean_cache: os.PathLike[str] | str,
    include_split: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    epoch: int = 1,
    base_seed: int = 20260813,
    scale_min: float = 0.005,
    scale_max: float = 0.020,
    expected_shape_count: int = 35_534,
    expected_point_count: int = 50_000,
    workers: int = 1,
    resume: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    """Build and atomically publish one epoch of PGD1 whole-cloud inputs.

    An interrupted build remains in ``.<output-name>.partial`` with its ledger
    in ``.<output-name>.state``.  Passing ``resume=True`` trusts only the
    fsynced ledger prefix and deterministically rebuilds uncommitted files.
    Each noisy SHA is computed from the serialized NPY payload before writing,
    so publication needs no second full-file hash pass.
    """

    clean_root = Path(clean_cache).resolve()
    if not clean_root.is_dir():
        raise FileNotFoundError(clean_root)
    split_path = Path(include_split).resolve()
    output = Path(output_dir)
    if not output.name:
        raise ValueError("output_dir must name a directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    output = output.parent.resolve() / output.name
    stage = output.parent / f".{output.name}.partial"
    state = output.parent / f".{output.name}.state"
    lock_path = output.parent / f".{output.name}.lock"
    epoch_index = _uint64(epoch, name="epoch", positive=True)
    seed = _uint64(base_seed, name="base_seed")
    low = _scale(scale_min, name="scale_min")
    high = _scale(scale_max, name="scale_max")
    if high <= low:
        raise ValueError("scale_max must be greater than scale_min")
    selected_count = _positive_integer(
        expected_shape_count, name="expected_shape_count"
    )
    point_count = _positive_integer(
        expected_point_count, name="expected_point_count"
    )
    worker_count = _positive_integer(workers, name="workers")
    if worker_count > 64:
        raise ValueError("workers must not exceed 64")
    if not isinstance(resume, bool):
        raise ValueError("resume must be a bool")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")

    clean_manifest, clean_records = _clean_manifest(clean_root)
    if clean_manifest.get("num_points") != point_count or clean_manifest.get(
        "array_contract"
    ) != {"container": "npy", "dtype": "float32", "shape": [point_count, 3]}:
        raise ValueError("clean-cache array contract differs from expected points")
    selected_ids, split_file_sha, declared_split_sha = _include_ids(split_path)
    if len(selected_ids) != selected_count:
        raise ValueError(
            f"include split count {len(selected_ids)} differs from "
            f"expected_shape_count {selected_count}"
        )
    unknown = sorted(set(selected_ids) - set(clean_records))
    if unknown:
        raise ValueError(f"include split IDs are absent from clean cache: {unknown[:3]!r}")
    shape_ids_sha = _canonical_sha256(selected_ids)
    split_sha = declared_split_sha or shape_ids_sha
    clean_content_sha = str(clean_manifest["content_sha256"])
    contract: dict[str, object] = {
        "format": "pcdenoise_pgd2_noisy_build_contract_v1",
        "epoch": epoch_index,
        "base_seed": seed,
        "scale_min": low,
        "scale_max": high,
        "num_points": point_count,
        "shape_count": selected_count,
        "shape_ids_sha256": shape_ids_sha,
        "split_sha256": split_sha,
        "source_clean_content_sha256": clean_content_sha,
        "include_split_file_sha256": split_file_sha,
    }

    with lock_path.open("a+b") as lock_stream:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another noisy-cache builder holds {lock_path}") from error
        if output.exists():
            if not resume:
                raise FileExistsError(output)
            if stage.exists():
                raise RuntimeError("both completed output and partial stage exist")
            manifest = _existing_completed(
                output,
                contract,
                clean_records=clean_records,
            )
            _cleanup_resume_state(state)
            return manifest
        contract_path = state / ".build_contract.json"
        progress_path = state / ".progress.jsonl"
        if stage.exists() or state.exists():
            if not resume:
                raise FileExistsError(
                    "partial build or resume state exists; rerun with "
                    f"resume=True: {stage}"
                )
            if not state.is_dir():
                raise RuntimeError("partial build is missing its resume-state directory")
            actual_contract = _json_from_bytes(
                _regular_file_bytes(contract_path), name="partial build contract"
            )
            if actual_contract != contract:
                raise ValueError("partial build contract differs from requested build")
            if not stage.exists():
                if _read_progress(progress_path):
                    raise RuntimeError("resume ledger exists without its partial output")
                stage.mkdir(mode=0o700)
        else:
            state.mkdir(mode=0o700)
            _write_json_exclusive(contract_path, contract)
            stage.mkdir(mode=0o700)

        records = _read_progress(progress_path)
        if len(records) > len(selected_ids):
            raise ValueError("progress ledger exceeds selected sample count")
        for index, record in enumerate(records):
            shape_id = selected_ids[index]
            expected_relative = f"shapenet/{shape_id}/noisy.npy"
            expected_seed = pgd2_noisy_seed(seed, epoch=epoch_index, shape_id=shape_id)
            expected_scale = float(
                np.random.default_rng(expected_seed).uniform(low, high)
            )
            if (
                set(record)
                != {
                    "shape_id",
                    "derived_seed",
                    "noise_scale",
                    "relative_path",
                    "noisy_file_bytes",
                    "noisy_sha256",
                    "source_clean_sha256",
                }
                or record.get("shape_id") != shape_id
                or record.get("relative_path") != expected_relative
                or record.get("derived_seed") != expected_seed
                or record.get("noise_scale") != expected_scale
                or _SHA256_RE.fullmatch(str(record.get("noisy_sha256"))) is None
                or record.get("source_clean_sha256")
                != clean_records[shape_id]["clean_sha256"]
            ):
                raise ValueError("progress ledger is not a valid selected-ID prefix")
            path = stage / expected_relative
            _validate_noisy_file(
                path,
                expected_point_count=point_count,
                expected_file_bytes=record.get("noisy_file_bytes"),
                expected_sha256=record.get("noisy_sha256"),
                label=f"committed noisy sample {shape_id}",
            )

        remaining = selected_ids[len(records) :]
        for shape_id in remaining:
            uncommitted = stage / f"shapenet/{shape_id}/noisy.npy"
            if uncommitted.exists():
                uncommitted.unlink()
            if uncommitted.parent.is_dir():
                for temporary in uncommitted.parent.glob(".noisy.npy.tmp-*"):
                    if temporary.is_file():
                        temporary.unlink()
        manifest_path = stage / "manifest.json"
        if manifest_path.exists() and len(records) != len(selected_ids):
            manifest_path.unlink()

        def generate(shape_id: str) -> dict[str, object]:
            source_record = clean_records[shape_id]
            clean_path = clean_root / str(source_record["relative_path"])
            clean = np.load(clean_path, allow_pickle=False)
            if (
                clean.dtype != np.float32
                or clean.shape != (point_count, 3)
                or not np.isfinite(clean).all()
            ):
                raise ValueError(f"invalid clean array for {shape_id}")
            derived_seed = pgd2_noisy_seed(seed, epoch=epoch_index, shape_id=shape_id)
            noisy, actual_scale = add_starter_laplace(
                clean,
                np.random.default_rng(derived_seed),
                scale_range=(low, high),
            )
            relative = f"shapenet/{shape_id}/noisy.npy"
            file_bytes, noisy_sha256 = _write_noisy(stage / relative, noisy)
            return {
                "shape_id": shape_id,
                "derived_seed": derived_seed,
                "noise_scale": actual_scale,
                "relative_path": relative,
                "noisy_file_bytes": file_bytes,
                "noisy_sha256": noisy_sha256,
                "source_clean_sha256": str(source_record["clean_sha256"]),
            }

        completed = len(records)
        batch_width = max(1, worker_count * 2)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for start in range(0, len(remaining), batch_width):
                shape_batch = remaining[start : start + batch_width]
                futures = [executor.submit(generate, shape_id) for shape_id in shape_batch]
                for shape_id, future in zip(shape_batch, futures):
                    record = future.result()
                    records.append(record)
                    _append_record(progress_path, record)
                    completed += 1
                    if progress is not None:
                        progress(completed, len(selected_ids), shape_id)

        manifest: dict[str, object] = {
            "format": PGD2_NOISY_CACHE_FORMAT,
            "format_version": PGD2_NOISY_CACHE_VERSION,
            "status": "completed",
            "epoch": epoch_index,
            "split_sha256": split_sha,
            "shape_count": len(selected_ids),
            "shape_ids": selected_ids,
            "num_points": point_count,
            "array_contract": {
                "container": "npy",
                "dtype": "float32",
                "shape": [point_count, 3],
                "finite": True,
                "index_order": "preserved_from_clean",
            },
            "noise_profile": "starter_laplace",
            "noise_policy": {
                "assignment": "independent_uniform_per_epoch_shape",
                "scale_min": low,
                "scale_max": high,
            },
            "base_seed": seed,
            "seed_derivation": _SEED_DERIVATION,
            "rng": {
                "library": "numpy",
                "numpy_version": np.__version__,
                "generator": "numpy.random.Generator",
                "bit_generator": type(np.random.default_rng(0).bit_generator).__name__,
                "stream_policy": "one_independent_stream_per_epoch_and_shape",
            },
            "source_clean_cache": {
                "format": clean_manifest.get("format"),
                "format_version": clean_manifest.get("format_version"),
                "content_sha256": clean_content_sha,
                "shape_count": clean_manifest.get("shape_count"),
                "num_points": point_count,
            },
            "include_split": {
                "file_sha256": split_file_sha,
                "declared_split_sha256": declared_split_sha,
                "sample_ids_sha256": shape_ids_sha,
                "sample_count": len(selected_ids),
            },
            "samples_sha256": _canonical_sha256(records),
            "samples": records,
        }
        manifest["content_sha256"] = _canonical_sha256(manifest)
        if manifest_path.exists():
            existing = _json_from_bytes(
                _regular_file_bytes(manifest_path), name="partial noisy manifest"
            )
            if existing != manifest:
                raise ValueError("partial completed manifest differs from regenerated manifest")
        else:
            _write_json_exclusive(manifest_path, manifest)
        expected_stage_files = {"manifest.json"} | {
            str(record["relative_path"]) for record in records
        }
        if (
            _regular_file_inventory(stage, label="partial PGD2 noisy cache")
            != expected_stage_files
        ):
            raise ValueError("partial PGD2 noisy cache inventory mismatch")
        stage_descriptor = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(stage_descriptor)
        finally:
            os.close(stage_descriptor)
        _rename_noreplace(stage, output)
        parent_descriptor = os.open(
            output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        _cleanup_resume_state(state)
        return manifest


__all__ = [
    "PGD2_NOISY_CACHE_FORMAT",
    "PGD2_NOISY_CACHE_VERSION",
    "build_pgd2_noisy_cache",
    "pgd2_noisy_seed",
]

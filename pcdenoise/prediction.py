"""Atomic batch prediction over competition-style noisy point clouds."""

from __future__ import annotations

import csv
import hashlib
import io
import inspect
import json
import math
import os
import re
import stat
import time
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import numpy as np

from pcdenoise.data.archive import (
    _cleanup_owned_stage,
    _create_owned_stage,
    _publish_owned_stage,
)
from pcdenoise.inference import (
    NOISE_ROUTE_MINIMUM_COVERAGE_RATIO,
    _owner_weighted_noise_summary,
    canonical_inference_options,
    denoise_cloud,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FileIdentity = tuple[int, int, int, int, int, int]
_ROUTE_MANIFEST_FIELDS = {
    "format",
    "format_version",
    "status",
    "input_contract",
    "score_semantics",
    "route_score",
    "prediction",
    "model_reference",
    "inference_options",
    "estimator_source",
    "aggregation_contract",
    "sample_count",
    "sample_ids",
    "samples",
    "artifacts",
    "content_sha256",
}
_ROUTE_STATISTIC_FIELDS = {
    "aggregation",
    "route_score_name",
    "route_score",
    "q50",
    "q75",
    "q90",
    "weighted_mad",
    "conditioning_maximum_scale",
    "conditioning_upper_saturation_fraction",
    "owned_point_count",
    "zero_owner_patch_count",
    "uncovered_count",
    "coverage_ratio",
    "patches",
}


@dataclass(frozen=True)
class NoisySample:
    sample_id: str
    path: Path
    file_identity: _FileIdentity | None = field(
        default=None,
        compare=False,
        repr=False,
    )


def _regular_file_identity(
    metadata: os.stat_result,
) -> _FileIdentity:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _validated_input_filename(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or "/" in value
        or "\\" in value
        or not value.endswith(".npy")
    ):
        raise ValueError(f"{name} must be a plain .npy basename")
    return value


def _sample_id(
    path: Path,
    root: Path,
    *,
    filename: str = "noisy.npy",
    label: str = "noisy sample",
) -> str:
    parts = path.relative_to(root).parts
    if "shapenet" not in parts:
        raise ValueError(f"{label} is outside shapenet layout: {path}")
    offset = parts.index("shapenet") + 1
    if len(parts) != offset + 3 or parts[-1] != filename:
        raise ValueError(f"invalid {label} layout: {path}")
    synset, model = parts[offset], parts[offset + 1]
    if (
        len(synset) != 8
        or not synset.isdigit()
        or not 28 <= len(model) <= 32
        or any(character not in "0123456789abcdef" for character in model)
    ):
        raise ValueError(f"invalid {label} ID: {synset}/{model}")
    return f"{synset}/{model}"


def _strict_discovered_input_path(
    path: Path,
    *,
    root: Path,
    label: str,
) -> tuple[Path, _FileIdentity]:
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} path must not contain symlinks: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    resolved_root = root.resolve(strict=True)
    resolved_path = path.resolve(strict=True)
    if resolved_root not in resolved_path.parents:
        raise ValueError(f"{label} escapes the input root: {path}")
    return path.absolute(), _regular_file_identity(metadata)


def _scan_inputs(
    root: Path | str,
    *,
    filename: object,
    filename_name: str,
    label: str,
) -> list[NoisySample]:
    actual_filename = _validated_input_filename(filename, name=filename_name)
    input_root = Path(root)
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    indexed: dict[
        str,
        tuple[Path, _FileIdentity],
    ] = {}
    for path in sorted(input_root.rglob(actual_filename)):
        sample_id = _sample_id(
            path,
            input_root,
            filename=actual_filename,
            label=label,
        )
        if sample_id in indexed:
            raise ValueError(f"duplicate {label} key: {sample_id}")
        indexed[sample_id] = _strict_discovered_input_path(
            path,
            root=input_root,
            label=label,
        )
    if not indexed:
        raise ValueError(f"input root contains no {actual_filename} samples")
    return [
        NoisySample(
            sample_id=sample_id,
            path=indexed[sample_id][0],
            file_identity=indexed[sample_id][1],
        )
        for sample_id in sorted(indexed)
    ]


def scan_noisy_inputs(root: Path | str) -> list[NoisySample]:
    return _scan_inputs(
        root,
        filename="noisy.npy",
        filename_name="input_filename",
        label="noisy sample",
    )


def _snapshot_regular(
    path: Path,
    *,
    expected_identity: _FileIdentity | None = None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"input is not a regular file: {path}")
        opened_identity = _regular_file_identity(metadata)
        if (
            expected_identity is not None
            and opened_identity != expected_identity
        ):
            raise ValueError(f"input changed after discovery: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        if _regular_file_identity(os.fstat(descriptor)) != opened_identity:
            raise ValueError(f"input changed while being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_input(
    path: Path,
    *,
    label: str,
    expected_identity: _FileIdentity | None = None,
) -> tuple[np.ndarray, str]:
    payload = _snapshot_regular(path, expected_identity=expected_identity)
    try:
        values = np.load(io.BytesIO(payload), allow_pickle=False)
    except (ValueError, OSError) as error:
        raise ValueError(f"invalid {label} NPY: {path}") from error
    if (
        values.dtype != np.float32
        or values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] != 3
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            f"{label} NPY must have finite shape (N,3) float32: {path}"
        )
    return (
        np.ascontiguousarray(values),
        hashlib.sha256(payload).hexdigest(),
    )


def _load_noisy(path: Path) -> tuple[np.ndarray, str]:
    return _load_input(path, label="noisy")


def _json_snapshot(
    value: Mapping[str, object],
    *,
    name: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must contain finite JSON-compatible values"
        ) from error
    result = json.loads(encoded)
    if not isinstance(result, dict):
        raise TypeError(f"{name} must be a mapping")
    return result


def _select_samples(
    samples: list[NoisySample],
    requested: Sequence[str] | None,
) -> list[NoisySample]:
    if requested is None:
        return samples
    if isinstance(requested, (str, bytes)) or not isinstance(
        requested, Sequence
    ):
        raise ValueError("sample_ids must be a sequence of IDs")
    selected_ids = list(requested)
    if not selected_ids:
        raise ValueError("sample_ids must not be empty")
    if not all(isinstance(sample_id, str) for sample_id in selected_ids):
        raise ValueError("sample_ids must contain strings")
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("sample_ids contains duplicates")
    indexed = {sample.sample_id: sample for sample in samples}
    unknown = sorted(set(selected_ids) - set(indexed))
    if unknown:
        raise ValueError(f"sample_ids contains unknown IDs: {unknown[:3]}")
    return [indexed[sample_id] for sample_id in sorted(selected_ids)]


def _save_npy(path: Path, points: np.ndarray) -> str:
    values = np.asarray(points)
    if (
        values.dtype != np.float32
        or values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] != 3
        or not np.isfinite(values).all()
    ):
        raise RuntimeError("prediction violated the (N,3) float32 contract")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(
            stream,
            np.ascontiguousarray(values),
            allow_pickle=False,
        )
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(_snapshot_regular(path)).hexdigest()


def _sha256_value(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical_digest(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _authoritative_input_inventory_sha256(
    samples: Sequence[Mapping[str, object]],
) -> str:
    inventory = [
        {
            "sample_id": sample["sample_id"],
            "authoritative_input_sha256": sample[
                "authoritative_input_sha256"
            ],
        }
        for sample in samples
    ]
    payload = json.dumps(
        inventory,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _no_duplicate_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_object(payload: bytes, *, name: str) -> dict[str, object]:
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_no_duplicate_object,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {name} JSON") from error
    if not isinstance(document, dict):
        raise ValueError(f"{name} must be a JSON object")
    return document


def _paths_are_disjoint(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return (
        first != second
        and first not in second.parents
        and second not in first.parents
    )


def _strict_sample_relative_path(
    prediction: Path,
    *,
    sample_id: object,
    relative_path: object,
) -> Path:
    if not isinstance(sample_id, str) or sample_id.count("/") != 1:
        raise ValueError("noise route sample_id is invalid")
    synset, model_id = sample_id.split("/")
    if (
        len(synset) != 8
        or not synset.isdigit()
        or not 28 <= len(model_id) <= 32
        or any(
            character not in "0123456789abcdef"
            for character in model_id
        )
    ):
        raise ValueError("noise route sample_id is invalid")
    expected = PurePosixPath(
        "shapenet",
        synset,
        model_id,
        "denoised.npy",
    )
    if not isinstance(relative_path, str) or relative_path != str(expected):
        raise ValueError("prediction relative_path differs from sample_id")

    current = prediction
    for part in expected.parts:
        current = current / part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("prediction output path must not contain symlinks")
    resolved = current.resolve(strict=True)
    if prediction not in resolved.parents:
        raise ValueError("prediction output escapes the prediction tree")
    return current


def _source_record(path: Path, *, module: str) -> dict[str, object]:
    absolute = path.resolve()
    payload = _snapshot_regular(absolute)
    return {
        "module": module,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _estimator_source_binding(model: object) -> dict[str, object]:
    model_class = type(model)
    source = inspect.getsourcefile(model_class)
    if source is None:
        raise ValueError("noise route model class has no inspectable source")
    model_record = _source_record(
        Path(source),
        module=f"{model_class.__module__}.{model_class.__qualname__}",
    )
    estimator = getattr(model, "noise_scale_estimator", None)
    estimator_class = type(estimator) if estimator is not None else model_class
    estimator_path = inspect.getsourcefile(estimator_class)
    if estimator_path is None:
        raise ValueError("noise scale estimator has no inspectable source")
    estimator_record = _source_record(
        Path(estimator_path),
        module=(
            f"{estimator_class.__module__}."
            f"{estimator_class.__qualname__}"
        ),
    )
    aggregation_record = _source_record(
        Path(__file__).with_name("inference.py"),
        module="pcdenoise.inference",
    )
    unsigned = {
        "model": model_record,
        "noise_scale_estimator": estimator_record,
        "aggregation": aggregation_record,
    }
    return {
        **unsigned,
        "content_sha256": _canonical_digest(unsigned),
    }


def _validate_route_model_reference(
    reference: Mapping[str, object],
) -> None:
    _sha256_value(
        reference.get("checkpoint_sha256"),
        name="model_reference.checkpoint_sha256",
    )
    _sha256_value(
        reference.get("config_sha256"),
        name="model_reference.config_sha256",
    )


def _write_route_sidecar(
    directory: Path,
    *,
    prediction_manifest: Mapping[str, object],
    prediction_manifest_sha256: str,
    model_reference: Mapping[str, object],
    estimator_source: Mapping[str, object],
    route_records: Sequence[Mapping[str, object]],
) -> None:
    csv_path = directory / "route_scores.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("sample_id", "route_score"),
            lineterminator="\n",
        )
        writer.writeheader()
        for record in route_records:
            statistics = record["statistics"]
            assert isinstance(statistics, Mapping)
            writer.writerow(
                {
                    "sample_id": record["sample_id"],
                    "route_score": format(
                        float(statistics["route_score"]),
                        ".17g",
                    ),
                }
            )
        stream.flush()
        os.fsync(stream.fileno())
    csv_sha256 = hashlib.sha256(_snapshot_regular(csv_path)).hexdigest()

    inference_options = {
        key: prediction_manifest[key]
        for key in (
            "patch_size",
            "seed_k",
            "patch_batch_size",
            "niters",
            "normalization_mode",
            "robust_quantile",
            "fusion_mode",
            "iteration_damping",
        )
    }
    samples = [dict(record) for record in route_records]
    unsigned = {
        "format": "pcdenoise_noise_route_diagnostic_v1",
        "format_version": 1,
        "status": "completed",
        "input_contract": (
            "noisy_points_only_no_clean_mesh_or_evaluator_metrics_v1"
        ),
        "score_semantics": "larger_means_more_estimated_noise",
        "route_score": "owner_point_weighted_q75_estimated_noise_scale",
        "prediction": {
            "format": prediction_manifest["format"],
            "format_version": prediction_manifest["format_version"],
            "inference_manifest_sha256": prediction_manifest_sha256,
        },
        "model_reference": dict(model_reference),
        "inference_options": inference_options,
        "estimator_source": dict(estimator_source),
        "aggregation_contract": {
            "owner_rule": "hard_best_owner_patch_v1",
            "point_weighting": "one_vote_per_owned_input_point_v1",
            "zero_owner_patch_weight": 0,
            "route_denominator": "sum_positive_owner_counts_v1",
            "minimum_coverage_ratio": (
                NOISE_ROUTE_MINIMUM_COVERAGE_RATIO
            ),
            "uncovered_route_policy": (
                "exclude_if_coverage_ge_minimum_else_fail_v1"
            ),
            "uncovered_output_policy": "preserve_normalized_input_point_v1",
            "quantile_method": "numpy_inverted_cdf_v1",
            "weighted_mad_center": "owner_point_weighted_q50",
            "upper_saturation_rule": (
                "estimated_scale_ge_float32_conditioning_maximum_v1"
            ),
        },
        "sample_count": len(route_records),
        "sample_ids": [str(record["sample_id"]) for record in route_records],
        "samples": samples,
        "artifacts": {"route_scores_csv_sha256": csv_sha256},
    }
    manifest = {**unsigned, "content_sha256": _canonical_digest(unsigned)}
    manifest_path = directory / "noise_route_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(
            manifest,
            stream,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _finite_number(
    value: object,
    *,
    name: str,
    allow_string: bool = False,
) -> float:
    if allow_string and isinstance(value, str):
        if not value or value.strip() != value:
            raise ValueError(f"{name} must be a finite number")
        candidate: object = value
    else:
        candidate = value
    if isinstance(candidate, bool) or not isinstance(
        candidate,
        (int, float, str) if allow_string else (int, float),
    ):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(candidate)
    except ValueError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _verify_route_statistics(
    value: object,
    *,
    expected_point_count: int,
    expected_num_patches: int,
    expected_uncovered_count: int,
    expected_coverage_ratio: float,
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _ROUTE_STATISTIC_FIELDS:
        raise ValueError("noise route statistic fields are invalid")
    if (
        value.get("aggregation")
        != "hard_best_owner_patch_point_weighted_v1"
        or value.get("route_score_name") != "q75"
    ):
        raise ValueError("noise route statistic identity is invalid")
    for field in (
        "route_score",
        "q50",
        "q75",
        "q90",
        "weighted_mad",
        "conditioning_maximum_scale",
        "conditioning_upper_saturation_fraction",
        "coverage_ratio",
    ):
        _finite_number(value.get(field), name=field)
    for field in (
        "owned_point_count",
        "zero_owner_patch_count",
        "uncovered_count",
    ):
        if isinstance(value.get(field), bool) or not isinstance(
            value.get(field), int
        ):
            raise ValueError(f"{field} must be an integer")
    if value["uncovered_count"] < 0:
        raise ValueError("noise route uncovered_count must be nonnegative")
    patches = value.get("patches")
    if not isinstance(patches, list) or not patches:
        raise ValueError("noise route patches must be a nonempty list")
    if len(patches) != expected_num_patches:
        raise ValueError("noise route patch ledger is incomplete")
    scales = []
    counts = []
    for patch_index, patch in enumerate(patches):
        if not isinstance(patch, dict) or set(patch) != {
            "patch_index",
            "estimated_noise_scale",
            "owner_point_count",
        }:
            raise ValueError("noise route patch fields are invalid")
        if patch.get("patch_index") != patch_index:
            raise ValueError("noise route patch order is invalid")
        scales.append(
            _finite_number(
                patch.get("estimated_noise_scale"),
                name="estimated_noise_scale",
            )
        )
        owner_count = patch.get("owner_point_count")
        if (
            isinstance(owner_count, bool)
            or not isinstance(owner_count, int)
            or owner_count < 0
        ):
            raise ValueError("owner_point_count must be nonnegative integer")
        counts.append(owner_count)
    maximum = _finite_number(
        value.get("conditioning_maximum_scale"),
        name="conditioning_maximum_scale",
    )
    recomputed = _owner_weighted_noise_summary(
        np.asarray(scales, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
        conditioning_maximum_scale=maximum,
        uncovered_count=value["uncovered_count"],
    )
    if recomputed != value:
        raise ValueError("noise route statistics do not match patch records")
    if (
        recomputed["owned_point_count"]
        + recomputed["uncovered_count"]
        != expected_point_count
    ):
        raise ValueError("noise route owner counts differ from point_count")
    if recomputed["uncovered_count"] != expected_uncovered_count:
        raise ValueError(
            "noise route uncovered_count differs from prediction"
        )
    if recomputed["coverage_ratio"] != expected_coverage_ratio:
        raise ValueError("noise route coverage differs from prediction")
    return recomputed


def _prediction_patch_contract(
    prediction_sample: Mapping[str, object],
    *,
    expected_point_count: int,
    inference_options: Mapping[str, object],
) -> dict[str, object]:
    details = prediction_sample.get("details")
    if not isinstance(details, dict):
        raise ValueError("prediction sample details must be an object")
    iterations = details.get("iterations")
    if (
        details.get("niters") != 1
        or details.get("fusion_mode") != "hard_best"
        or not isinstance(iterations, list)
        or len(iterations) != 1
        or not isinstance(iterations[0], dict)
    ):
        raise ValueError("prediction route iteration contract is invalid")
    iteration = iterations[0]
    if (
        iteration.get("iteration") != 1
        or iteration.get("residual_damping") != 1.0
        or iteration.get("point_count") != expected_point_count
        or iteration.get("patch_size") != inference_options["patch_size"]
        or iteration.get("seed_k") != inference_options["seed_k"]
        or iteration.get("patch_batch_size")
        != inference_options["patch_batch_size"]
        or iteration.get("fusion_mode") != "hard_best"
        or iteration.get("zero_weight_fallback_count") != 0
    ):
        raise ValueError("prediction patch details are invalid")
    num_patches = iteration.get("num_patches")
    if (
        isinstance(num_patches, bool)
        or not isinstance(num_patches, int)
        or num_patches <= 0
    ):
        raise ValueError("prediction num_patches is invalid")
    uncovered_count = iteration.get("uncovered_count")
    if (
        isinstance(uncovered_count, bool)
        or not isinstance(uncovered_count, int)
        or not 0 <= uncovered_count < expected_point_count
    ):
        raise ValueError("prediction uncovered_count is invalid")
    coverage_ratio = _finite_number(
        iteration.get("coverage_fraction"),
        name="prediction coverage_fraction",
    )
    expected_coverage_ratio = (
        expected_point_count - uncovered_count
    ) / expected_point_count
    if coverage_ratio != expected_coverage_ratio:
        raise ValueError(
            "prediction coverage_fraction differs from uncovered_count"
        )
    if coverage_ratio < NOISE_ROUTE_MINIMUM_COVERAGE_RATIO:
        raise ValueError(
            "prediction route coverage is below the frozen minimum"
        )
    return {
        "num_patches": num_patches,
        "uncovered_count": uncovered_count,
        "coverage_ratio": coverage_ratio,
    }


def verify_noise_route_diagnostic(
    sidecar_dir: Path | str,
    *,
    prediction_dir: Path | str,
) -> dict[str, object]:
    """Strictly verify one input-only route sidecar and its prediction bytes."""

    sidecar = Path(sidecar_dir).resolve()
    prediction = Path(prediction_dir).resolve()
    if not sidecar.is_dir() or not prediction.is_dir():
        raise FileNotFoundError("sidecar and prediction directories must exist")
    actual_files = {path.name for path in sidecar.iterdir()}
    if actual_files != {"noise_route_manifest.json", "route_scores.csv"}:
        raise ValueError("noise route sidecar files are invalid")

    manifest_payload = _snapshot_regular(
        sidecar / "noise_route_manifest.json"
    )
    csv_payload = _snapshot_regular(sidecar / "route_scores.csv")
    document = _load_json_object(
        manifest_payload,
        name="noise route manifest",
    )
    if set(document) != _ROUTE_MANIFEST_FIELDS:
        raise ValueError("noise route manifest fields are invalid")
    claimed_content = _sha256_value(
        document.get("content_sha256"),
        name="noise route content_sha256",
    )
    unsigned = dict(document)
    unsigned.pop("content_sha256")
    if _canonical_digest(unsigned) != claimed_content:
        raise ValueError("noise route content SHA is invalid")
    if (
        document.get("format") != "pcdenoise_noise_route_diagnostic_v1"
        or type(document.get("format_version")) is not int
        or document.get("format_version") != 1
        or document.get("status") != "completed"
        or document.get("input_contract")
        != "noisy_points_only_no_clean_mesh_or_evaluator_metrics_v1"
        or document.get("score_semantics")
        != "larger_means_more_estimated_noise"
        or document.get("route_score")
        != "owner_point_weighted_q75_estimated_noise_scale"
    ):
        raise ValueError("noise route manifest identity is invalid")

    prediction_manifest_path = prediction / "inference_manifest.json"
    prediction_payload = _snapshot_regular(prediction_manifest_path)
    prediction_sha256 = hashlib.sha256(prediction_payload).hexdigest()
    prediction_manifest = _load_json_object(
        prediction_payload,
        name="prediction manifest",
    )
    if (
        prediction_manifest.get("format") != "pcdenoise_prediction_v1"
        or type(prediction_manifest.get("format_version")) is not int
        or prediction_manifest.get("format_version") != 1
        or prediction_manifest.get("status") != "completed"
    ):
        raise ValueError("prediction manifest identity is invalid")
    prediction_binding = document.get("prediction")
    expected_prediction_binding = {
        "format": prediction_manifest.get("format"),
        "format_version": prediction_manifest.get("format_version"),
        "inference_manifest_sha256": prediction_sha256,
    }
    if prediction_binding != expected_prediction_binding:
        raise ValueError("noise route prediction binding is invalid")
    if document.get("model_reference") != prediction_manifest.get(
        "model_reference"
    ):
        raise ValueError("noise route model reference differs from prediction")
    model_reference = document.get("model_reference")
    if not isinstance(model_reference, dict):
        raise ValueError("noise route model reference must be an object")
    _validate_route_model_reference(model_reference)

    expected_options = {
        key: prediction_manifest.get(key)
        for key in (
            "patch_size",
            "seed_k",
            "patch_batch_size",
            "niters",
            "normalization_mode",
            "robust_quantile",
            "fusion_mode",
            "iteration_damping",
        )
    }
    if document.get("inference_options") != expected_options:
        raise ValueError("noise route inference options differ from prediction")
    if expected_options["niters"] != 1:
        raise ValueError("noise route sidecar requires niters=1")
    if expected_options["fusion_mode"] != "hard_best":
        raise ValueError("noise route sidecar requires hard_best fusion")

    estimator_source = document.get("estimator_source")
    if not isinstance(estimator_source, dict) or set(estimator_source) != {
        "model",
        "noise_scale_estimator",
        "aggregation",
        "content_sha256",
    }:
        raise ValueError("noise route estimator source fields are invalid")
    for label in ("model", "noise_scale_estimator", "aggregation"):
        record = estimator_source[label]
        if not isinstance(record, dict) or set(record) != {
            "module",
            "source_sha256",
        }:
            raise ValueError("noise route source record fields are invalid")
        if not isinstance(record["module"], str) or not record["module"]:
            raise ValueError("noise route source module is invalid")
        _sha256_value(record["source_sha256"], name="source_sha256")
    estimator_unsigned = {
        "model": estimator_source["model"],
        "noise_scale_estimator": estimator_source[
            "noise_scale_estimator"
        ],
        "aggregation": estimator_source["aggregation"],
    }
    if estimator_source["content_sha256"] != _canonical_digest(
        estimator_unsigned
    ):
        raise ValueError("noise route estimator source SHA is invalid")

    aggregation_contract = document.get("aggregation_contract")
    if aggregation_contract != {
        "owner_rule": "hard_best_owner_patch_v1",
        "point_weighting": "one_vote_per_owned_input_point_v1",
        "zero_owner_patch_weight": 0,
        "route_denominator": "sum_positive_owner_counts_v1",
        "minimum_coverage_ratio": NOISE_ROUTE_MINIMUM_COVERAGE_RATIO,
        "uncovered_route_policy": (
            "exclude_if_coverage_ge_minimum_else_fail_v1"
        ),
        "uncovered_output_policy": "preserve_normalized_input_point_v1",
        "quantile_method": "numpy_inverted_cdf_v1",
        "weighted_mad_center": "owner_point_weighted_q50",
        "upper_saturation_rule": (
            "estimated_scale_ge_float32_conditioning_maximum_v1"
        ),
    }:
        raise ValueError("noise route aggregation contract is invalid")

    prediction_samples = prediction_manifest.get("samples")
    prediction_ids = prediction_manifest.get("sample_ids")
    samples = document.get("samples")
    sample_count = document.get("sample_count")
    if (
        not isinstance(prediction_samples, list)
        or not isinstance(prediction_ids, list)
        or not isinstance(samples, list)
        or isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or prediction_manifest.get("sample_count") != sample_count
        or sample_count != len(prediction_ids)
        or document.get("sample_ids") != prediction_ids
        or len(samples) != sample_count
        or len(samples) != len(prediction_samples)
    ):
        raise ValueError("noise route sample ledger differs from prediction")
    if not all(isinstance(sample_id, str) for sample_id in prediction_ids):
        raise ValueError("prediction sample IDs must be strings")
    if len(set(prediction_ids)) != sample_count:
        raise ValueError("prediction sample IDs must be unique")
    if not all(isinstance(sample, dict) for sample in prediction_samples):
        raise ValueError("prediction sample records must be objects")
    if not all(isinstance(sample, dict) for sample in samples):
        raise ValueError("noise route sample records must be objects")
    if [sample.get("sample_id") for sample in prediction_samples] != (
        prediction_ids
    ):
        raise ValueError("prediction sample records differ from sample_ids")
    if [sample.get("sample_id") for sample in samples] != prediction_ids:
        raise ValueError("noise route sample records differ from sample_ids")

    route_scores: list[tuple[str, float]] = []
    for sample, prediction_sample in zip(samples, prediction_samples):
        if not isinstance(sample, dict) or set(sample) != {
            "sample_id",
            "point_count",
            "input_sha256",
            "output_sha256",
            "statistics",
        }:
            raise ValueError("noise route sample fields are invalid")
        if not isinstance(prediction_sample, dict):
            raise ValueError("prediction sample record is invalid")
        for field in (
            "sample_id",
            "point_count",
            "input_sha256",
            "output_sha256",
        ):
            if sample[field] != prediction_sample.get(field):
                raise ValueError(
                    f"noise route sample {field} differs from prediction"
                )
        _sha256_value(sample["input_sha256"], name="input_sha256")
        expected_output_sha256 = _sha256_value(
            sample["output_sha256"],
            name="output_sha256",
        )
        output_path = _strict_sample_relative_path(
            prediction,
            sample_id=sample["sample_id"],
            relative_path=prediction_sample.get("relative_path"),
        )
        if hashlib.sha256(_snapshot_regular(output_path)).hexdigest() != (
            expected_output_sha256
        ):
            raise ValueError("prediction output bytes differ from route binding")
        point_count = sample["point_count"]
        if (
            isinstance(point_count, bool)
            or not isinstance(point_count, int)
            or point_count <= 0
        ):
            raise ValueError("noise route point_count is invalid")
        patch_contract = _prediction_patch_contract(
            prediction_sample,
            expected_point_count=point_count,
            inference_options=expected_options,
        )
        statistics = _verify_route_statistics(
            sample["statistics"],
            expected_point_count=point_count,
            expected_num_patches=int(patch_contract["num_patches"]),
            expected_uncovered_count=int(
                patch_contract["uncovered_count"]
            ),
            expected_coverage_ratio=float(
                patch_contract["coverage_ratio"]
            ),
        )
        route_scores.append(
            (str(sample["sample_id"]), float(statistics["route_score"]))
        )

    claimed_csv_sha = document.get("artifacts")
    actual_csv_sha = hashlib.sha256(csv_payload).hexdigest()
    if claimed_csv_sha != {"route_scores_csv_sha256": actual_csv_sha}:
        raise ValueError("noise route CSV SHA is invalid")
    try:
        reader = csv.DictReader(
            io.StringIO(csv_payload.decode("utf-8"), newline=""),
            strict=True,
        )
        if reader.fieldnames != ["sample_id", "route_score"]:
            raise ValueError("noise route CSV header is invalid")
        csv_scores = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("noise route CSV row is malformed")
            csv_scores.append(
                (
                    row["sample_id"],
                    _finite_number(
                        row["route_score"],
                        name="route_score",
                        allow_string=True,
                    ),
                )
            )
    except (UnicodeError, csv.Error) as error:
        raise ValueError("invalid noise route CSV") from error
    if csv_scores != route_scores:
        raise ValueError("noise route CSV values/order differ from manifest")
    return document


def run_prediction(
    model: object,
    *,
    input_root: Path | str,
    input_filename: str = "noisy.npy",
    authoritative_input_root: Path | str | None = None,
    authoritative_input_filename: str = "noisy.npy",
    output_dir: Path | str,
    patch_size: int = 1000,
    seed_k: float = 6,
    patch_batch_size: int = 5,
    niters: int = 1,
    normalization_mode: str = "noisy_max",
    robust_quantile: float | None = None,
    fusion_mode: str = "hard_best",
    iteration_damping: float = 1.0,
    sample_ids: Sequence[str] | None = None,
    model_reference: Mapping[str, object],
    noise_route_diagnostic_dir: Path | str | None = None,
) -> dict[str, object]:
    """Denoise selected inputs and atomically publish one complete directory."""

    options = canonical_inference_options(
        normalization_mode=normalization_mode,
        robust_quantile=robust_quantile,
        fusion_mode=fusion_mode,
        iteration_damping=iteration_damping,
    )
    actual_input_filename = _validated_input_filename(
        input_filename,
        name="input_filename",
    )
    if (
        actual_input_filename != "noisy.npy"
        and authoritative_input_root is None
    ):
        raise ValueError(
            "authoritative_input_root is required for nonlegacy input_filename"
        )
    actual_authoritative_filename = "noisy.npy"
    if authoritative_input_root is not None:
        actual_authoritative_filename = _validated_input_filename(
            authoritative_input_filename,
            name="authoritative_input_filename",
        )
    legacy_input_contract = (
        actual_input_filename == "noisy.npy"
        and authoritative_input_root is None
    )
    if legacy_input_contract:
        all_samples = scan_noisy_inputs(input_root)
        model_input_label = "noisy"
    else:
        all_samples = _scan_inputs(
            input_root,
            filename=actual_input_filename,
            filename_name="input_filename",
            label="model input",
        )
        model_input_label = "model input"
    authoritative_by_id: dict[str, NoisySample] | None = None
    if authoritative_input_root is not None:
        authoritative_samples = _scan_inputs(
            authoritative_input_root,
            filename=actual_authoritative_filename,
            filename_name="authoritative_input_filename",
            label="authoritative input",
        )
        if [sample.sample_id for sample in authoritative_samples] != [
            sample.sample_id for sample in all_samples
        ]:
            raise ValueError("authoritative input ID universe mismatch")
        authoritative_by_id = {
            sample.sample_id: sample for sample in authoritative_samples
        }
    samples = _select_samples(all_samples, sample_ids)
    reference = _json_snapshot(model_reference, name="model_reference")
    capture_noise_route = noise_route_diagnostic_dir is not None
    route_destination = None
    estimator_source = None
    if capture_noise_route:
        if (
            isinstance(niters, bool)
            or not isinstance(niters, Integral)
            or int(niters) != 1
        ):
            raise ValueError(
                "noise route diagnostic currently requires niters=1"
            )
        if options["fusion_mode"] != "hard_best":
            raise ValueError(
                "noise route diagnostic requires fusion_mode='hard_best'"
            )
        output_destination = Path(output_dir).resolve()
        route_destination = Path(noise_route_diagnostic_dir).resolve()
        if not _paths_are_disjoint(output_destination, route_destination):
            raise ValueError(
                "noise route diagnostic must be outside the prediction tree"
            )
        _validate_route_model_reference(reference)
        estimator_source = _estimator_source_binding(model)
    stage = _create_owned_stage(output_dir)
    route_stage = None
    if route_destination is not None:
        try:
            route_stage = _create_owned_stage(route_destination)
        except BaseException:
            _cleanup_owned_stage(stage)
            raise
    started = time.perf_counter()
    sample_records: list[dict[str, object]] = []
    route_records: list[dict[str, object]] = []
    try:
        for sample in samples:
            noisy, input_sha256 = _load_input(
                sample.path,
                label=model_input_label,
                expected_identity=sample.file_identity,
            )
            authoritative_input_sha256 = None
            if authoritative_by_id is not None:
                authoritative, authoritative_input_sha256 = _load_input(
                    authoritative_by_id[sample.sample_id].path,
                    label="authoritative input",
                    expected_identity=authoritative_by_id[
                        sample.sample_id
                    ].file_identity,
                )
                if authoritative.shape != noisy.shape:
                    raise ValueError(
                        f"{sample.sample_id}: authoritative input shape "
                        "mismatch"
                    )
            sample_started = time.perf_counter()
            denoised, details = denoise_cloud(
                model,
                noisy,
                patch_size=patch_size,
                seed_k=seed_k,
                patch_batch_size=patch_batch_size,
                niters=niters,
                normalization_mode=str(options["normalization_mode"]),
                robust_quantile=options["robust_quantile"],
                fusion_mode=str(options["fusion_mode"]),
                iteration_damping=float(options["iteration_damping"]),
                capture_noise_route=capture_noise_route,
            )
            route_statistics = None
            if capture_noise_route:
                route_statistics = details.pop(
                    "noise_route_diagnostic",
                    None,
                )
                if not isinstance(route_statistics, dict):
                    raise RuntimeError(
                        "noise route diagnostic was not returned by inference"
                    )
            if denoised.shape != noisy.shape:
                raise RuntimeError(
                    f"{sample.sample_id}: output point count/shape changed"
                )
            synset, model_id = sample.sample_id.split("/")
            relative_path = (
                Path("shapenet")
                / synset
                / model_id
                / "denoised.npy"
            )
            output_sha256 = _save_npy(
                stage.path / relative_path,
                denoised.astype(np.float32, copy=False),
            )
            sample_record = {
                "sample_id": sample.sample_id,
                "point_count": len(noisy),
                "input_sha256": input_sha256,
                "output_sha256": output_sha256,
                "relative_path": relative_path.as_posix(),
                "elapsed_seconds": time.perf_counter() - sample_started,
                "details": details,
            }
            if authoritative_input_sha256 is not None:
                sample_record["authoritative_input_sha256"] = (
                    authoritative_input_sha256
                )
            sample_records.append(sample_record)
            if capture_noise_route:
                assert route_statistics is not None
                route_records.append(
                    {
                        "sample_id": sample.sample_id,
                        "point_count": len(noisy),
                        "input_sha256": input_sha256,
                        "output_sha256": output_sha256,
                        "statistics": route_statistics,
                    }
                )

        elapsed = time.perf_counter() - started
        manifest = {
            "format": "pcdenoise_prediction_v1",
            "format_version": 1,
            "status": "completed",
            "sample_count": len(samples),
            "sample_ids": [sample.sample_id for sample in samples],
            "patch_size": int(patch_size),
            "seed_k": float(seed_k),
            "patch_batch_size": int(patch_batch_size),
            "niters": int(niters),
            "normalization_mode": options["normalization_mode"],
            "robust_quantile": options["robust_quantile"],
            "fusion_mode": options["fusion_mode"],
            "iteration_damping": options["iteration_damping"],
            "model_reference": reference,
            "elapsed_seconds": elapsed,
            "samples": sample_records,
        }
        if actual_input_filename != "noisy.npy" or (
            authoritative_by_id is not None
        ):
            manifest["input_filename"] = actual_input_filename
        if authoritative_by_id is not None:
            manifest["authoritative_input_filename"] = (
                actual_authoritative_filename
            )
            manifest["authoritative_input_inventory_sha256"] = (
                _authoritative_input_inventory_sha256(sample_records)
            )
        manifest_path = stage.path / "inference_manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(
                manifest,
                stream,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if route_stage is not None:
            assert estimator_source is not None
            _write_route_sidecar(
                route_stage.path,
                prediction_manifest=manifest,
                prediction_manifest_sha256=hashlib.sha256(
                    _snapshot_regular(manifest_path)
                ).hexdigest(),
                model_reference=reference,
                estimator_source=estimator_source,
                route_records=route_records,
            )
            # Re-read every staged artifact through the same strict verifier
            # used after publication.  This catches writer/schema drift before
            # either directory becomes visible at its destination.
            verify_noise_route_diagnostic(
                route_stage.path,
                prediction_dir=stage.path,
            )
        published = _publish_owned_stage(stage)
        published_route = (
            _publish_owned_stage(route_stage)
            if route_stage is not None
            else None
        )
    except BaseException:
        _cleanup_owned_stage(stage)
        if route_stage is not None:
            _cleanup_owned_stage(route_stage)
        raise
    result = {
        "output_dir": str(published),
        "sample_count": len(samples),
        "sample_ids": [sample.sample_id for sample in samples],
        "elapsed_seconds": elapsed,
        "model_reference": reference,
    }
    if published_route is not None:
        result["noise_route_diagnostic_dir"] = str(published_route)
    return result


__all__ = [
    "NoisySample",
    "run_prediction",
    "scan_noisy_inputs",
    "verify_noise_route_diagnostic",
]

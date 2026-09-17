#!/usr/bin/env python3
"""Evaluate denoised point clouds with exact competition CD/P2S semantics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import stat
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdenoise.metrics.chamfer import as_float64_points, chamfer_distance
from pcdenoise.data.archive import (
    _cleanup_owned_stage,
    _create_owned_stage,
    _publish_owned_stage,
)
from pcdenoise.data.validation_cache import (
    VALIDATION_CACHE_V3_FORMAT,
    verify_validation_cache,
)
from pcdenoise.metrics.competition import (
    aggregate_sample_scores,
    score_sample_metrics,
)
from pcdenoise.metrics.p2s import point_to_surface_distance
from pcdenoise.training.logger import TensorBoardLogger

try:
    import point_cloud_utils as pcu
except ImportError:  # pragma: no cover - guarded by the environment checker.
    pcu = None


EVALUATION_FORMAT = "pcdenoise_evaluation_v1"
EVALUATION_VERSION = 1


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _snapshot_regular_file(path: Path) -> tuple[bytes, str]:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"input must be a regular non-symlink file: {path}")
    payload = path.read_bytes()
    after = path.lstat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(payload) != after.st_size:
        raise ValueError(f"input changed while it was read: {path}")
    return payload, hashlib.sha256(payload).hexdigest()


def _write_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _is_unverified_v3_cache(root: Path) -> bool:
    manifest_path = root / "manifest.json"
    if not os.path.lexists(manifest_path):
        return False
    try:
        payload, _ = _snapshot_regular_file(manifest_path)
        manifest = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return False
    return (
        isinstance(manifest, dict)
        and manifest.get("format") == VALIDATION_CACHE_V3_FORMAT
    )


def _prediction_inventory(
    predictions: dict[str, Path],
    sample_ids: list[str],
) -> dict[str, object]:
    records = []
    for sample_id in sample_ids:
        _, sha256 = _snapshot_regular_file(predictions[sample_id])
        records.append({"shape_id": sample_id, "sha256": sha256})
    return {
        "sample_count": len(sample_ids),
        "shape_ids": sample_ids,
        "content_sha256": _canonical_digest(records),
    }


def _sample_key(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    parts = relative.parts
    if "shapenet" in parts:
        start = parts.index("shapenet") + 1
        if len(parts) < start + 3:
            raise ValueError(f"cannot derive synset/model key from {path}")
        return f"{parts[start]}/{parts[start + 1]}"
    if (
        len(parts) >= 4
        and parts[-2:] == ("models", "model_normalized.obj")
    ):
        return f"{parts[-4]}/{parts[-3]}"
    if len(parts) < 3:
        raise ValueError(f"cannot derive synset/model key from {path}")
    return f"{parts[-3]}/{parts[-2]}"


def _scan(root: Path, filename: str) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    samples: dict[str, Path] = {}
    for path in sorted(root.rglob(filename)):
        key = _sample_key(path, root)
        if key in samples:
            raise ValueError(f"duplicate {filename} sample key: {key}")
        samples[key] = path
    return samples


def _scan_meshes(root: Path) -> dict[str, Path]:
    return _scan(root, "model_normalized.obj")


def _load_points(path: Path, name: str) -> np.ndarray:
    points = np.load(path, allow_pickle=False)
    return as_float64_points(points, name)


def _evaluate_one(
    task: tuple[
        str,
        str,
        str,
        str,
        str | None,
        tuple[tuple[float, float, float], float] | None,
    ],
) -> dict[str, object]:
    (
        sample_id,
        pred_name,
        clean_name,
        noisy_name,
        mesh_name,
        mesh_normalization,
    ) = task
    prediction = _load_points(Path(pred_name), "prediction")
    clean = _load_points(Path(clean_name), "clean")
    noisy = _load_points(Path(noisy_name), "noisy")
    if len(prediction) != len(noisy):
        raise ValueError(
            f"{sample_id}: prediction has {len(prediction)} points, "
            f"noisy input has {len(noisy)}"
        )

    cd_pred = chamfer_distance(prediction, clean, normalize=True)
    cd_noisy = chamfer_distance(noisy, clean, normalize=True)
    if mesh_name is None:
        return score_sample_metrics(
            sample_id=sample_id,
            cd_pred=cd_pred,
            cd_noisy=cd_noisy,
        )

    if pcu is None:
        raise RuntimeError("point-cloud-utils is required for strict P2S")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module="point_cloud_utils",
        )
        vertices, faces = pcu.load_mesh_vf(mesh_name)
    normalize_reference: np.ndarray | None = clean
    if mesh_normalization is not None:
        center_values, scale_value = mesh_normalization
        center = np.asarray(center_values, dtype=np.float64)
        scale = float(scale_value)
        if (
            center.shape != (3,)
            or not np.isfinite(center).all()
            or not np.isfinite(scale)
            or scale <= 0.0
        ):
            raise ValueError(
                f"{sample_id}: invalid cached mesh normalization"
            )
        vertices = (
            np.asarray(vertices, dtype=np.float64) - center
        ) / scale
    p2s_pred = point_to_surface_distance(
        prediction,
        vertices,
        faces,
        normalize_reference=normalize_reference,
    )
    p2s_noisy = point_to_surface_distance(
        noisy,
        vertices,
        faces,
        normalize_reference=normalize_reference,
    )
    return score_sample_metrics(
        sample_id=sample_id,
        cd_pred=cd_pred,
        cd_noisy=cd_noisy,
        p2s_pred=p2s_pred,
        p2s_noisy=p2s_noisy,
    )


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    fieldnames = [
        "sample_id",
        "synset",
        "missing",
        "cd_pred",
        "cd_noisy",
        "cd_score",
        "p2s_pred",
        "p2s_noisy",
        "p2s_score",
        "total_score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})
        stream.flush()
        os.fsync(stream.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--noisy-dir", type=Path, required=True)
    parser.add_argument("--mesh-dir", type=Path)
    parser.add_argument("--pred-filename", default="denoised.npy")
    parser.add_argument("--gt-filename", default="clean.npy")
    parser.add_argument("--noisy-filename", default="noisy.npy")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--validation-cache-manifest",
        type=Path,
        help="preflight a v2/v3 validation cache before strict evaluation",
    )
    parser.add_argument(
        "--validation-split",
        type=Path,
        help="bind preflight to the exact validation.json input",
    )
    parser.add_argument(
        "--expected-validation-content-sha256",
        help="externally pinned v2/v3 validation-cache content SHA256",
    )
    parser.add_argument(
        "--trust-validation-cache-files",
        action="store_true",
        help=(
            "validate the pinned cache manifest and split without re-reading "
            "and hashing every clean/noisy/mesh file; intended for repeated "
            "local candidate screening after one strict cache verification"
        ),
    )
    parser.add_argument(
        "--sample-ids",
        type=Path,
        help="optional UTF-8 file with one exact <synset>/<model> ID per line",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if os.path.lexists(args.output_dir):
        raise FileExistsError(f"refusing to overwrite output: {args.output_dir}")
    source_payload, evaluator_source_sha256 = _snapshot_regular_file(
        Path(__file__).resolve()
    )
    cache_manifest: dict[str, object] | None = None
    cache_manifest_file_sha256: str | None = None
    if args.validation_cache_manifest is not None:
        if args.validation_cache_manifest.name != "manifest.json":
            raise ValueError(
                "--validation-cache-manifest must name manifest.json"
            )
        cache_root = args.validation_cache_manifest.parent.resolve()
        if (
            args.gt_dir.resolve() != cache_root
            or args.noisy_dir.resolve() != cache_root
        ):
            raise ValueError(
                "preflight cache must be both --gt-dir and --noisy-dir"
            )
        cache_payload, cache_manifest_file_sha256 = _snapshot_regular_file(
            args.validation_cache_manifest
        )
        cache_manifest = verify_validation_cache(
            cache_root,
            mesh_root=args.mesh_dir,
            validation_split=args.validation_split,
            expected_content_sha256=(
                args.expected_validation_content_sha256
            ),
            verify_files=not args.trust_validation_cache_files,
        )
        verified_payload, verified_file_sha256 = _snapshot_regular_file(
            args.validation_cache_manifest
        )
        if (
            verified_payload != cache_payload
            or verified_file_sha256 != cache_manifest_file_sha256
        ):
            raise ValueError(
                "validation cache manifest changed while it was verified"
            )
    elif (
        args.validation_split is not None
        or args.expected_validation_content_sha256 is not None
        or args.trust_validation_cache_files
    ):
        raise ValueError(
            "validation preflight options require "
            "--validation-cache-manifest"
        )
    elif args.mesh_dir is not None and _is_unverified_v3_cache(args.gt_dir):
        raise ValueError(
            "v3 validation cache with --mesh-dir requires "
            "--validation-cache-manifest"
        )

    predictions = _scan(args.pred_dir, args.pred_filename)
    clean = _scan(args.gt_dir, args.gt_filename)
    noisy = _scan(args.noisy_dir, args.noisy_filename)
    if set(clean) != set(noisy):
        raise ValueError("clean and noisy sample sets differ")
    if args.sample_ids is not None:
        requested = [
            line.strip()
            for line in args.sample_ids.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        if not requested:
            raise ValueError("--sample-ids file is empty")
        if len(set(requested)) != len(requested):
            raise ValueError("--sample-ids contains duplicate IDs")
        unknown = sorted(set(requested) - set(clean))
        if unknown:
            raise ValueError(
                f"--sample-ids contains unknown IDs: {unknown[:3]}"
            )
        selected = set(requested)
        clean = {
            sample_id: path
            for sample_id, path in clean.items()
            if sample_id in selected
        }
        noisy = {
            sample_id: path
            for sample_id, path in noisy.items()
            if sample_id in selected
        }
        predictions = {
            sample_id: path
            for sample_id, path in predictions.items()
            if sample_id in selected
        }
    meshes = _scan_meshes(args.mesh_dir) if args.mesh_dir is not None else {}
    mesh_normalizations: dict[
        str,
        tuple[tuple[float, float, float], float],
    ] = {}
    if (
        cache_manifest is not None
        and cache_manifest.get("format") == VALIDATION_CACHE_V3_FORMAT
    ):
        raw_samples = cache_manifest.get("samples")
        if not isinstance(raw_samples, list):
            raise ValueError("v3 validation cache samples are invalid")
        for record in raw_samples:
            if not isinstance(record, dict):
                raise ValueError(
                    "v3 validation cache sample record is invalid"
                )
            sample_id = record.get("shape_id")
            raw_center = record.get("normalization_center")
            raw_scale = record.get("normalization_scale")
            if (
                not isinstance(sample_id, str)
                or not isinstance(raw_center, list)
                or len(raw_center) != 3
            ):
                raise ValueError(
                    "v3 validation cache normalization metadata is invalid"
                )
            center = tuple(float(value) for value in raw_center)
            scale = float(raw_scale)
            mesh_normalizations[sample_id] = (center, scale)
    if args.mesh_dir is not None:
        if args.sample_ids is not None:
            meshes = {
                sample_id: path
                for sample_id, path in meshes.items()
                if sample_id in clean
            }
        missing_meshes = sorted(set(clean) - set(meshes))
        if missing_meshes:
            raise ValueError(f"missing meshes: {missing_meshes[:3]}")
        if (
            cache_manifest is not None
            and cache_manifest.get("format")
            == VALIDATION_CACHE_V3_FORMAT
        ):
            missing_normalizations = sorted(
                set(clean) - set(mesh_normalizations)
            )
            if missing_normalizations:
                raise ValueError(
                    "missing v3 mesh normalizations: "
                    f"{missing_normalizations[:3]}"
                )

    valid_ids = sorted(set(clean) & set(predictions))
    prediction_inventory = _prediction_inventory(predictions, valid_ids)
    tasks = [
        (
            sample_id,
            str(predictions[sample_id]),
            str(clean[sample_id]),
            str(noisy[sample_id]),
            str(meshes[sample_id]) if args.mesh_dir is not None else None,
            mesh_normalizations.get(sample_id),
        )
        for sample_id in valid_ids
    ]
    worker_count = (
        args.workers
        if args.workers > 0
        else min(multiprocessing.cpu_count(), 16)
    )
    if worker_count > 1 and len(tasks) > 1:
        with multiprocessing.Pool(worker_count) as pool:
            rows = pool.map(_evaluate_one, tasks)
    else:
        rows = [_evaluate_one(task) for task in tasks]

    summary = aggregate_sample_scores(
        rows,
        expected_sample_ids=sorted(clean),
        require_p2s=args.mesh_dir is not None,
    )
    repeated_inventory = _prediction_inventory(predictions, valid_ids)
    if repeated_inventory != prediction_inventory:
        raise ValueError("prediction inventory changed during evaluation")

    stage = _create_owned_stage(args.output_dir)
    try:
        metrics_path = stage.path / "metrics_per_sample.csv"
        _write_csv(metrics_path, summary["per_sample"])
        summary_path = stage.path / "metrics_summary.json"
        _write_text(
            summary_path,
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )
        with TensorBoardLogger(stage.path) as tensorboard:
            for key in (
                "mean_cd_score",
                "mean_p2s_score",
                "total_score",
                "mean_cd_pred",
                "mean_cd_noisy",
                "mean_p2s_pred",
                "mean_p2s_noisy",
            ):
                value = summary.get(key)
                if value is not None:
                    tensorboard.log_scalar(
                        f"evaluation/{key}",
                        value,
                        step=0,
                    )
            for index, row in enumerate(summary["per_sample"]):
                tensorboard.log_scalar(
                    "samples/total_score",
                    row["total_score"],
                    step=index,
                )
                tensorboard.log_scalar(
                    "samples/cd_score",
                    row["cd_score"],
                    step=index,
                )
                if row["p2s_score"] is not None:
                    tensorboard.log_scalar(
                        "samples/p2s_score",
                        row["p2s_score"],
                        step=index,
                    )
            for index, sample_id in enumerate(valid_ids[:3]):
                tag = sample_id.replace("/", "_")
                tensorboard.log_point_cloud(
                    f"pointcloud/{tag}/noisy",
                    _load_points(noisy[sample_id], "noisy"),
                    step=index,
                )
                tensorboard.log_point_cloud(
                    f"pointcloud/{tag}/prediction",
                    _load_points(predictions[sample_id], "prediction"),
                    step=index,
                )
                tensorboard.log_point_cloud(
                    f"pointcloud/{tag}/target",
                    _load_points(clean[sample_id], "clean"),
                    step=index,
                )
            tensorboard.flush()

        repeated_source, repeated_source_sha256 = _snapshot_regular_file(
            Path(__file__).resolve()
        )
        if (
            repeated_source != source_payload
            or repeated_source_sha256 != evaluator_source_sha256
        ):
            raise ValueError("evaluator source changed during evaluation")
        validation_binding = None
        if cache_manifest is not None:
            validation_binding = {
                "format": cache_manifest["format"],
                "format_version": cache_manifest["format_version"],
                "noise_profile": cache_manifest.get("noise_profile"),
                "content_sha256": cache_manifest["content_sha256"],
                "manifest_file_sha256": cache_manifest_file_sha256,
                "files_verified": not args.trust_validation_cache_files,
            }
        evaluation_manifest = {
            "format": EVALUATION_FORMAT,
            "format_version": EVALUATION_VERSION,
            "completed": True,
            "sample_count": len(clean),
            "sample_ids": sorted(clean),
            "valid_sample_count": len(valid_ids),
            "require_p2s": args.mesh_dir is not None,
            "validation_cache": validation_binding,
            "prediction_inventory": prediction_inventory,
            "artifacts": {
                "metrics_per_sample_csv_sha256": (
                    _snapshot_regular_file(metrics_path)[1]
                ),
                "metrics_summary_json_sha256": (
                    _snapshot_regular_file(summary_path)[1]
                ),
            },
            "evaluator_source_sha256": evaluator_source_sha256,
        }
        evaluation_manifest["content_sha256"] = _canonical_digest(
            evaluation_manifest
        )
        _write_text(
            stage.path / "evaluation_manifest.json",
            json.dumps(
                evaluation_manifest,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )
        _publish_owned_stage(stage)
    except BaseException:
        _cleanup_owned_stage(stage)
        raise
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "sample_count",
                    "valid_count",
                    "missing_count",
                    "mean_cd_score",
                    "mean_p2s_score",
                    "total_score",
                    "mean_cd_pred",
                    "mean_cd_noisy",
                    "mean_p2s_pred",
                    "mean_p2s_noisy",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

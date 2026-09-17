#!/usr/bin/env python3
"""Build and fully verify the deterministic PGD clean-surface cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdenoise.data.surface_cache import (
    build_surface_cache,
    build_surface_cache_v2,
    verify_surface_cache,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--select-count", type=int, default=4096)
    parser.add_argument("--num-points", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=(
            "parallel mesh workers; 8-16 is recommended and peak host "
            "memory grows with workers, mesh size, point count, and views"
        ),
    )
    parser.add_argument(
        "--cache-format",
        choices=("v1", "v2"),
        default="v1",
        help="v1 preserves the legacy area-only single-view cache",
    )
    parser.add_argument(
        "--profile",
        choices=(
            "area_surface_v1",
            "paired_vertex_replacement_v1",
            "starter_vertex_mix_v1",
            "paired_face_vertex_replacement_v2",
            "starter_face_vertex_mix_v2",
        ),
        default="area_surface_v1",
    )
    parser.add_argument("--vertex-sample-count", type=int, default=0)
    parser.add_argument("--view-count", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--expected-train-split-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-split-manifest-sha256",
        required=True,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=64,
        help="emit one stderr progress line after this many completed shapes",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip the default full output-file, mesh-hash and split audit",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="exclusively create a machine-readable build/audit report",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(8 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(path: Path, report: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    if arguments.cache_format == "v1" and (
        arguments.profile != "area_surface_v1"
        or arguments.vertex_sample_count != 0
        or arguments.view_count != 1
    ):
        raise ValueError(
            "v1 requires profile=area_surface_v1, "
            "vertex_sample_count=0, and view_count=1"
        )
    if arguments.cache_format == "v2" and arguments.workers > 16:
        print(
            "[surface-cache] warning: workers above the recommended 8-16 "
            "range increase peak host-memory demand",
            file=sys.stderr,
            flush=True,
        )

    def progress(completed: int, total: int, shape_id: str) -> None:
        if (
            completed == 1
            or completed == total
            or completed % arguments.progress_every == 0
        ):
            elapsed = time.monotonic() - start
            rate = completed / elapsed if elapsed > 0.0 else 0.0
            print(
                f"[surface-cache] {completed}/{total} "
                f"elapsed={elapsed:.1f}s rate={rate:.2f} shape/s "
                f"last={shape_id}",
                file=sys.stderr,
                flush=True,
            )

    shared_build_arguments = {
        "mesh_root": arguments.mesh_root,
        "train_split": arguments.train_split,
        "output_dir": arguments.output_dir,
        "select_count": arguments.select_count,
        "num_points": arguments.num_points,
        "seed": arguments.seed,
        "workers": arguments.workers,
        "expected_train_split_sha256": (
            arguments.expected_train_split_sha256
        ),
        "expected_split_manifest_sha256": (
            arguments.expected_split_manifest_sha256
        ),
        "progress": progress,
    }
    if arguments.cache_format == "v2":
        manifest = build_surface_cache_v2(
            **shared_build_arguments,
            profile=arguments.profile,
            vertex_sample_count=arguments.vertex_sample_count,
            view_count=arguments.view_count,
        )
    else:
        manifest = build_surface_cache(**shared_build_arguments)
    build_seconds = time.monotonic() - start
    verified = False
    verify_seconds = 0.0
    if not arguments.skip_verify:
        verify_start = time.monotonic()
        verify_surface_cache(
            arguments.output_dir,
            mesh_root=arguments.mesh_root,
            train_split=arguments.train_split,
            expected_train_split_sha256=(
                arguments.expected_train_split_sha256
            ),
            expected_split_manifest_sha256=(
                arguments.expected_split_manifest_sha256
            ),
            expected_content_sha256=manifest["content_sha256"],
            verify_files=True,
        )
        verify_seconds = time.monotonic() - verify_start
        verified = True
        print(
            f"[surface-cache] full verification passed in "
            f"{verify_seconds:.1f}s",
            file=sys.stderr,
            flush=True,
        )
    manifest_path = arguments.output_dir / "manifest.json"
    if arguments.cache_format == "v2":
        array_file_bytes = sum(
            int(view["clean_file_bytes"])
            for record in manifest["samples"]
            for view in record["views"]
        )
    else:
        array_file_bytes = sum(
            int(record["clean_file_bytes"])
            for record in manifest["samples"]
        )
    report = {
        "status": "READY" if verified else "BUILT_NOT_FULLY_VERIFIED",
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_seconds": build_seconds,
        "verify_seconds": verify_seconds,
        "total_seconds": time.monotonic() - start,
        "output_dir": os.fspath(arguments.output_dir),
        "shape_count": manifest["shape_count"],
        "num_points": manifest["num_points"],
        "seed": manifest["seed"],
        "workers": arguments.workers,
        "array_file_bytes": array_file_bytes,
        "content_sha256": manifest["content_sha256"],
        "samples_sha256": manifest["samples_sha256"],
        "manifest_file_sha256": _sha256(manifest_path),
        "train_split_sha256": manifest["train_split_sha256"],
        "split_manifest_sha256": manifest["split_manifest_sha256"],
        "full_verify": verified,
    }
    if arguments.cache_format == "v2":
        report.update(
            {
                "cache_format": "v2",
                "profile": manifest["sampling"]["profile"],
                "vertex_sample_count": manifest["sampling"][
                    "vertex_sample_count"
                ],
                "view_count": manifest["view_count"],
                "worker_guidance": (
                    "8-16 workers recommended; peak host memory scales "
                    "with mesh size, point count, view count, and workers"
                ),
            }
        )
    if arguments.report_json is not None:
        _write_report(arguments.report_json, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

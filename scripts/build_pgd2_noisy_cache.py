#!/usr/bin/env python3
"""Build one deterministic, resumable PGD2 Laplace noisy-cache epoch."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdenoise.data.pgd2_noisy_cache import build_pgd2_noisy_cache


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-cache", type=Path, required=True)
    parser.add_argument(
        "--include-split",
        type=Path,
        required=True,
        help="strict JSON or one-shape-ID-per-line PGD2 training selection",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--scale-min", type=float, default=0.005)
    parser.add_argument("--scale-max", type=float, default=0.020)
    parser.add_argument("--expected-shape-count", type=int, default=35_534)
    parser.add_argument("--expected-point-count", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=64)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    started = time.monotonic()

    def progress(completed: int, total: int, shape_id: str) -> None:
        if completed == 1 or completed == total or completed % arguments.progress_every == 0:
            elapsed = time.monotonic() - started
            rate = completed / elapsed if elapsed > 0.0 else 0.0
            print(
                f"[pgd2-noisy] epoch={arguments.epoch} {completed}/{total} "
                f"elapsed={elapsed:.1f}s rate={rate:.2f} shape/s last={shape_id}",
                file=sys.stderr,
                flush=True,
            )

    manifest = build_pgd2_noisy_cache(
        clean_cache=arguments.clean_cache,
        include_split=arguments.include_split,
        output_dir=arguments.output_dir,
        epoch=arguments.epoch,
        base_seed=arguments.seed,
        scale_min=arguments.scale_min,
        scale_max=arguments.scale_max,
        expected_shape_count=arguments.expected_shape_count,
        expected_point_count=arguments.expected_point_count,
        workers=arguments.workers,
        resume=arguments.resume,
        progress=progress,
    )
    summary = {
        key: manifest[key]
        for key in (
            "status",
            "epoch",
            "split_sha256",
            "shape_count",
            "num_points",
            "noise_profile",
            "noise_policy",
            "base_seed",
            "samples_sha256",
            "content_sha256",
        )
    }
    summary["output_dir"] = str(arguments.output_dir.resolve())
    summary["elapsed_seconds"] = time.monotonic() - started
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

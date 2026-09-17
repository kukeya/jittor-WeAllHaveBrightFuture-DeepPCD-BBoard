#!/usr/bin/env python3
"""Publish hardlinked PGD1-output/clean pairs for README PGD2 training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdenoise.data.pgd2_paired_cache import build_pgd2_paired_cache


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-cache", type=Path, required=True)
    parser.add_argument(
        "--input-cache",
        type=Path,
        required=True,
        help="completed per-epoch PGD2 noisy cache used as PGD1 input",
    )
    parser.add_argument(
        "--shard-root",
        type=Path,
        required=True,
        help="merged PGD1 prediction root with inference_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-point-count", type=int, default=50_000)
    parser.add_argument(
        "--expected-pgd1-checkpoint-sha256",
        required=True,
        help="authoritative frozen PGD1 checkpoint SHA256",
    )
    parser.add_argument(
        "--expected-pgd1-config-sha256",
        required=True,
        help="authoritative frozen PGD1 canonical config SHA256",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    manifest = build_pgd2_paired_cache(
        clean_cache=arguments.clean_cache,
        noisy_cache=arguments.input_cache,
        pgd1_output=arguments.shard_root,
        output_dir=arguments.output_dir,
        expected_point_count=arguments.expected_point_count,
        expected_pgd1_checkpoint_sha256=(
            arguments.expected_pgd1_checkpoint_sha256
        ),
        expected_pgd1_config_sha256=arguments.expected_pgd1_config_sha256,
    )
    print(
        json.dumps(
            {
                "format": manifest["format"],
                "status": manifest["status"],
                "sample_count": manifest["sample_count"],
                "point_count": manifest["point_count"],
                "storage_mode": manifest["layout"]["storage_mode"],
                "content_sha256": manifest["content_sha256"],
                "pgd1": manifest["pgd1"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Map official B-test sample names to the IDs used by PGD inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np


_OFFICIAL_RE = re.compile(r"^btest_([0-9]{6})$")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--expected-points", type=int, default=50_000)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan(
    root: Path,
    *,
    expected_count: int,
    expected_points: int,
) -> list[tuple[int, Path, str]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    indexed: dict[int, tuple[Path, str]] = {}
    for source in sorted(root.rglob("noisy.npy")):
        relative = source.relative_to(root).parts
        if (
            len(relative) != 4
            or relative[0] != "shapenet"
            or relative[1] != "00000000"
            or relative[3] != "noisy.npy"
        ):
            raise ValueError(f"unexpected B-test path: {source}")
        match = _OFFICIAL_RE.fullmatch(relative[2])
        if match is None:
            raise ValueError(f"invalid official B-test ID: {relative[2]}")
        ordinal = int(match.group(1))
        if ordinal in indexed:
            raise ValueError(f"duplicate B-test ordinal: {ordinal}")
        points = np.load(source, mmap_mode="r", allow_pickle=False)
        if points.dtype != np.float32 or points.shape != (expected_points, 3):
            raise ValueError(
                f"invalid B-test array contract for {source}: "
                f"{points.dtype} {points.shape}"
            )
        if not np.isfinite(points).all():
            raise ValueError(f"non-finite B-test input: {source}")
        indexed[ordinal] = (source, _sha256(source))

    expected_ordinals = list(range(1, expected_count + 1))
    if sorted(indexed) != expected_ordinals:
        raise ValueError(
            "B-test IDs must be the contiguous range "
            f"btest_000001..btest_{expected_count:06d}"
        )
    return [
        (ordinal, indexed[ordinal][0], indexed[ordinal][1])
        for ordinal in expected_ordinals
    ]


def main() -> int:
    args = _parser().parse_args()
    if args.expected_count <= 0 or args.expected_points <= 0:
        raise ValueError("expected-count and expected-points must be positive")
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    records = _scan(
        args.input_root,
        expected_count=args.expected_count,
        expected_points=args.expected_points,
    )

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_root.name}.build-",
            dir=args.output_root.parent,
        )
    )
    mappings: list[dict[str, object]] = []
    try:
        for ordinal, source, source_sha256 in records:
            official_id = f"btest_{ordinal:06d}"
            mapped_id = f"b{ordinal:031x}"
            destination = (
                stage
                / "shapenet"
                / "00000000"
                / mapped_id
                / "noisy.npy"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            destination_sha256 = _sha256(destination)
            if destination_sha256 != source_sha256:
                raise RuntimeError(f"copy digest mismatch: {official_id}")
            mappings.append(
                {
                    "ordinal": ordinal,
                    "official_id": f"00000000/{official_id}",
                    "mapped_id": f"00000000/{mapped_id}",
                    "noisy_sha256": source_sha256,
                }
            )

        manifest = {
            "format": "pcdenoise_b_test_mapping_v1",
            "status": "completed",
            "sample_count": len(mappings),
            "point_count": args.expected_points,
            "dtype": "float32",
            "input_root": str(args.input_root.resolve()),
            "mappings": mappings,
        }
        manifest_path = stage / "mapping_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, args.output_root)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

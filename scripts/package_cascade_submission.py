#!/usr/bin/env python3
"""Validate final cascade predictions and build the B-track result.zip.

The cascade launcher keeps the mapped internal IDs used during inference:
``00000000/b<31 hexadecimal digits>``.  The competition archive uses
``00000000/btest_<six decimal digits>``.  This script performs only that
lossless ID conversion, validates every array, and writes a deterministic ZIP.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np


_MAPPED_RE = re.compile(r"^b([0-9a-f]{31})$")
_OFFICIAL_RE = re.compile(r"^btest_([0-9]{6})$")
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--expected-points", type=int, default=50_000)
    return parser


def _official_name(source_name: str) -> str:
    official = _OFFICIAL_RE.fullmatch(source_name)
    if official is not None:
        ordinal = int(official.group(1))
    else:
        mapped = _MAPPED_RE.fullmatch(source_name)
        if mapped is None:
            raise ValueError(f"unsupported prediction sample ID: {source_name}")
        ordinal = int(mapped.group(1), 16)
    if ordinal <= 0 or ordinal > 999_999:
        raise ValueError(f"sample ordinal is outside the B-track range: {ordinal}")
    return f"btest_{ordinal:06d}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect(root: Path, expected_points: int) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for source in sorted(root.glob("shapenet/*/*/denoised.npy")):
        relative = source.relative_to(root).parts
        if len(relative) != 4 or relative[0] != "shapenet":
            raise ValueError(f"unexpected prediction path: {source}")
        group, source_name = relative[1], relative[2]
        official_name = _official_name(source_name)
        archive_name = f"shapenet/{group}/{official_name}/denoised.npy"
        if archive_name in seen:
            raise ValueError(f"duplicate B-track sample: {archive_name}")
        array = np.load(source, allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (expected_points, 3):
            raise ValueError(
                f"invalid array contract for {source}: {array.dtype} {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite prediction: {source}")
        seen.add(archive_name)
        entries.append((archive_name, source))
    return sorted(entries)


def _write_zip(entries: list[tuple[str, Path]], output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for archive_name, source in entries:
            info = zipfile.ZipInfo(archive_name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())


def _verify_zip(output: Path, expected_names: list[str], expected_points: int) -> None:
    with zipfile.ZipFile(output, "r") as archive:
        if archive.namelist() != expected_names:
            raise RuntimeError("result.zip member order or names are invalid")
        if archive.testzip() is not None:
            raise RuntimeError("result.zip CRC validation failed")
        for name in expected_names:
            array = np.load(io.BytesIO(archive.read(name)), allow_pickle=False)
            if array.dtype != np.float32 or array.shape != (expected_points, 3):
                raise RuntimeError(f"invalid array after ZIP round trip: {name}")
            if not np.isfinite(array).all():
                raise RuntimeError(f"non-finite array after ZIP round trip: {name}")


def main() -> int:
    args = _parser().parse_args()
    if not args.prediction_root.is_dir():
        raise FileNotFoundError(args.prediction_root)
    if args.expected_count <= 0 or args.expected_points <= 0:
        raise ValueError("expected-count and expected-points must be positive")
    entries = _collect(args.prediction_root, args.expected_points)
    if len(entries) != args.expected_count:
        raise ValueError(
            f"expected {args.expected_count} predictions, found {len(entries)}"
        )
    _write_zip(entries, args.output_zip)
    names = [name for name, _ in entries]
    _verify_zip(args.output_zip, names, args.expected_points)
    summary = {
        "format": "pcdenoise_b_cascade_submission_v1",
        "status": "completed",
        "prediction_root": str(args.prediction_root.resolve()),
        "output_zip": str(args.output_zip.resolve()),
        "entry_count": len(entries),
        "point_count": args.expected_points,
        "dtype": "float32",
        "first_member": names[0],
        "last_member": names[-1],
        "zip_sha256": _sha256(args.output_zip),
    }
    if args.manifest_output is not None:
        if args.manifest_output.exists():
            raise FileExistsError(args.manifest_output)
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

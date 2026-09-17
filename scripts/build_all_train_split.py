#!/usr/bin/env python3
"""Build an immutable all-training split from a prepared stratified split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdenoise.data.split import (
    SHA256_RE,
    _validated_ids,
    build_all_train_split,
    write_split_artifacts,
)


_TEST_KEYS = frozenset(
    (
        "format",
        "format_version",
        "split",
        "count",
        "shape_ids",
        "archive_sha256",
    )
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-split",
        type=Path,
        required=True,
        help="source split directory or its manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _source_paths(source: Path) -> tuple[Path, Path]:
    if source.is_dir():
        return source / "manifest.json", source / "test.json"
    if source.is_file():
        return source, source.with_name("test.json")
    raise FileNotFoundError(source)


def _read_json(path: Path, *, label: str) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be readable JSON: {path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object")
    return document


def _validated_test_bundle(document: dict) -> tuple[list[str], str]:
    try:
        if set(document) != _TEST_KEYS:
            raise ValueError("unexpected test fields")
        if (
            document["format"] != "pcdenoise_shape_ids_v1"
            or isinstance(document["format_version"], bool)
            or document["format_version"] != 1
            or document["split"] != "test"
        ):
            raise ValueError("invalid test metadata")
        raw_ids = document["shape_ids"]
        if not isinstance(raw_ids, list):
            raise ValueError("test shape_ids must be a list")
        shape_ids = _validated_ids(raw_ids)
        if (
            raw_ids != shape_ids
            or isinstance(document["count"], bool)
            or document["count"] != len(shape_ids)
        ):
            raise ValueError("test shape ordering/count is invalid")
        archive_sha256 = document["archive_sha256"]
        if (
            not isinstance(archive_sha256, str)
            or not SHA256_RE.fullmatch(archive_sha256)
        ):
            raise ValueError("invalid test archive SHA256")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("source test split is invalid") from error
    return shape_ids, archive_sha256


def main() -> int:
    args = _parser().parse_args()
    manifest_path, test_path = _source_paths(args.source_split)
    source_manifest = _read_json(manifest_path, label="source split")
    try:
        all_train = build_all_train_split(source_manifest)
    except ValueError as error:
        raise ValueError("source split manifest is invalid") from error
    test_ids, test_archive_sha256 = _validated_test_bundle(
        _read_json(test_path, label="source test split")
    )
    paths = write_split_artifacts(
        all_train,
        args.output_dir,
        test_shape_ids=test_ids,
        test_archive_sha256=test_archive_sha256,
    )
    report = {
        "algorithm": all_train["algorithm"],
        "source_split_sha256": all_train["source_split_sha256"],
        "split_sha256": all_train["split_sha256"],
        "counts": all_train["counts"],
        "output_dir": str(paths["manifest"].parent.resolve()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

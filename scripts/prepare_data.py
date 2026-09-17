#!/usr/bin/env python3
"""Validate, inventory, split, and optionally extract official archives."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdenoise.data.archive import (
    ArchiveValidationError,
    _cleanup_owned_stage,
    _create_owned_stage,
    _path_lexists,
    _publish_owned_stage,
    _verify_archive_hash,
    extract_test_zip,
    extract_train_tar,
    inspect_test_zip,
    inspect_train_tar,
)
from pcdenoise.data.split import (
    build_stratified_split,
    load_official_split_from_starter,
    write_split_artifacts,
)


@dataclass(frozen=True)
class _PublishedArtifact:
    path: Path
    device: int
    inode: int
    is_directory: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-archive",
        type=Path,
        default=Path("data/dataset_train.tar.gz"),
    )
    parser.add_argument(
        "--test-archive",
        type=Path,
        default=Path("data/dataset_test_noisy.zip"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/prepared")
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/splits/seed_20260726"),
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="scan metadata and hashes but do not extract or create split files",
    )
    parser.add_argument(
        "--inventory-json",
        type=Path,
        help="optionally write the combined report; refuses overwrite",
    )
    parser.add_argument(
        "--official-starter",
        type=Path,
        help="optional Starter ZIP used only for an official-split audit",
    )
    return parser


def _write_new_json(
    path: Path,
    value: object,
    *,
    before_publish: Callable[[Path, int, int, bool], None] | None = None,
) -> None:
    if _path_lexists(path):
        raise FileExistsError(f"refusing to overwrite report: {path}")
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.resolve()
    destination = parent / path.name
    if _path_lexists(destination):
        raise FileExistsError(f"refusing to overwrite report: {destination}")
    prefix = f".{path.name}.pcdenoise-stage-"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=prefix,
        dir=os.fspath(parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        metadata = os.lstat(temporary)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(
            metadata.st_mode
        ):
            raise RuntimeError(
                f"report staging path is no longer a regular file: {temporary}"
            )
        if before_publish is not None:
            before_publish(
                destination,
                metadata.st_dev,
                metadata.st_ino,
                False,
            )
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite report: {destination}"
            ) from error
        published = os.lstat(destination)
        if (
            published.st_dev != metadata.st_dev
            or published.st_ino != metadata.st_ino
        ):
            destination.unlink()
            raise RuntimeError(
                f"report staging identity changed during publication: {path}"
            )
    finally:
        if _path_lexists(temporary):
            if (
                temporary.parent.resolve() != parent
                or not temporary.name.startswith(prefix)
            ):
                raise RuntimeError(
                    f"refusing unsafe temporary cleanup: {temporary}"
                )
            temporary.unlink()


def _preflight_absent(path: Path, label: str) -> None:
    if _path_lexists(path):
        raise FileExistsError(f"refusing to overwrite {label}: {path}")


def _prospective_path(path: Path) -> Path:
    if not path.name:
        raise ValueError(f"output path must name a file or directory: {path}")
    return path.parent.resolve() / path.name


def _capture_published_artifact(
    path: Path,
    *,
    is_directory: bool,
    expected_device: int | None = None,
    expected_inode: int | None = None,
) -> _PublishedArtifact:
    absolute = _prospective_path(path)
    if (expected_device is None) != (expected_inode is None):
        raise ValueError("published artifact identity must be complete")
    if expected_device is not None and expected_inode is not None:
        return _PublishedArtifact(
            path=absolute,
            device=expected_device,
            inode=expected_inode,
            is_directory=is_directory,
        )
    metadata = os.lstat(absolute)
    if is_directory:
        valid_type = stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(
            metadata.st_mode
        )
    else:
        valid_type = stat.S_ISREG(metadata.st_mode) and not stat.S_ISLNK(
            metadata.st_mode
        )
    if not valid_type:
        raise RuntimeError(
            f"published artifact has unexpected type: {absolute}"
        )
    return _PublishedArtifact(
        path=absolute,
        device=metadata.st_dev,
        inode=metadata.st_ino,
        is_directory=is_directory,
    )


def _rollback_published_artifact(artifact: _PublishedArtifact) -> None:
    if not _path_lexists(artifact.path):
        return
    metadata = os.lstat(artifact.path)
    if (
        metadata.st_dev != artifact.device
        or metadata.st_ino != artifact.inode
    ):
        raise RuntimeError(
            f"refusing to remove replaced artifact: {artifact.path}"
        )
    if artifact.is_directory:
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(
            metadata.st_mode
        ):
            raise RuntimeError(
                f"refusing to remove non-directory artifact: {artifact.path}"
            )
        shutil.rmtree(artifact.path)
    else:
        if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(
            metadata.st_mode
        ):
            raise RuntimeError(
                f"refusing to remove non-file artifact: {artifact.path}"
            )
        artifact.path.unlink()


def _preflight_output_paths(args: argparse.Namespace) -> None:
    if not args.inventory_only:
        _preflight_absent(args.output_dir, "output root")
        _preflight_absent(args.split_dir, "split output")
    if args.inventory_json is not None:
        _preflight_absent(args.inventory_json, "report")

    if args.inventory_only:
        return
    destinations = [
        ("output root", _prospective_path(args.output_dir)),
        ("split output", _prospective_path(args.split_dir)),
    ]
    if args.inventory_json is not None:
        destinations.append(
            ("report", _prospective_path(args.inventory_json))
        )
    for index, (left_label, left) in enumerate(destinations):
        for right_label, right in destinations[index + 1 :]:
            if (
                left == right
                or left in right.parents
                or right in left.parents
            ):
                raise ValueError(
                    f"{left_label} and {right_label} overlap: "
                    f"{left} vs {right}"
                )


def _require_same_inventory(
    label: str,
    expected: dict,
    actual: dict,
) -> None:
    if actual != expected:
        raise ArchiveValidationError(
            f"{label} archive changed after initial validation"
        )


def _write_completion_manifest(
    stage: Path,
    train_inventory: dict,
    test_inventory: dict,
    split: dict,
) -> None:
    completion = {
        "format_version": 1,
        "status": "complete",
        "train_archive_sha256": train_inventory["archive_sha256"],
        "test_archive_sha256": test_inventory["archive_sha256"],
        "train_obj_count": train_inventory["obj_count"],
        "test_npy_count": test_inventory["npy_count"],
        "split_sha256": split["split_sha256"],
    }
    path = stage / "completion_manifest.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(completion, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    args = _parser().parse_args()
    _preflight_output_paths(args)
    train_inventory = inspect_train_tar(args.train_archive)
    test_inventory = inspect_test_zip(args.test_archive)
    split = build_stratified_split(
        train_inventory["shape_ids"],
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    report = {
        "format_version": 1,
        "train_inventory": train_inventory,
        "test_inventory": test_inventory,
        "primary_split": split,
    }
    if args.official_starter is not None:
        report["official_split_audit"] = load_official_split_from_starter(
            args.official_starter,
            known_shape_ids=train_inventory["shape_ids"],
        )
    if args.inventory_only:
        if args.inventory_json is not None:
            _write_new_json(args.inventory_json, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    output_stage = _create_owned_stage(args.output_dir)
    published_artifacts = []

    def register_publication(
        path: Path,
        device: int,
        inode: int,
        is_directory: bool,
    ) -> None:
        artifact = _capture_published_artifact(
            path,
            is_directory=is_directory,
            expected_device=device,
            expected_inode=inode,
        )
        published_artifacts.append(artifact)

    try:
        extracted_train = extract_train_tar(
            args.train_archive, output_stage.path / "train"
        )
        _require_same_inventory(
            "train", train_inventory, extracted_train
        )
        extracted_test = extract_test_zip(
            args.test_archive, output_stage.path / "test"
        )
        _require_same_inventory("test", test_inventory, extracted_test)
        _write_completion_manifest(
            output_stage.path,
            train_inventory,
            test_inventory,
            split,
        )
        write_split_artifacts(
            split,
            args.split_dir,
            test_shape_ids=test_inventory["shape_ids"],
            test_archive_sha256=test_inventory["archive_sha256"],
            before_publish=register_publication,
        )
        if args.inventory_json is not None:
            _write_new_json(
                args.inventory_json,
                report,
                before_publish=register_publication,
            )
        _verify_archive_hash(args.train_archive, train_inventory)
        _verify_archive_hash(args.test_archive, test_inventory)
        _publish_owned_stage(output_stage)
    except BaseException as error:
        cleanup_errors = []
        try:
            _cleanup_owned_stage(output_stage)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        for artifact in reversed(published_artifacts):
            try:
                _rollback_published_artifact(artifact)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            messages = "; ".join(str(item) for item in cleanup_errors)
            raise RuntimeError(
                f"transaction rollback was incomplete: {messages}"
            ) from error
        raise
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

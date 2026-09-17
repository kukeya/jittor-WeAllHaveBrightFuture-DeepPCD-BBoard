#!/usr/bin/env python3
"""Prepare one immutable A+B mesh index and all-training split.

The operation does not copy mesh payloads.  It creates hard links in a new
mesh root, preserving the official A/B files while presenting one directory
layout to the unchanged surface-cache and training code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdenoise.data.archive import (
    _cleanup_owned_stage,
    _create_owned_stage,
    _publish_owned_stage,
)
from pcdenoise.data.split import (
    build_all_train_split,
    build_stratified_split,
    write_split_artifacts,
)


_B_LINE_RE = re.compile(
    r"^shapenet/00000000/btrain_([0-9a-f]{24})$"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-mesh-root", type=Path, required=True)
    parser.add_argument("--a-train-split", type=Path, required=True)
    parser.add_argument("--b-mesh-root", type=Path, required=True)
    parser.add_argument("--b-train-list", type=Path, required=True)
    parser.add_argument("--b-validation-list", type=Path, required=True)
    parser.add_argument("--output-mesh-root", type=Path, required=True)
    parser.add_argument("--output-split-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_a_bundle(path: Path) -> tuple[list[str], dict[str, object]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(document, dict)
        or document.get("format") != "pcdenoise_shape_ids_v1"
        or document.get("format_version") != 1
        or document.get("split") != "train"
        or not isinstance(document.get("shape_ids"), list)
    ):
        raise ValueError(f"invalid A train split: {path}")
    shape_ids = document["shape_ids"]
    if (
        not shape_ids
        or any(not isinstance(item, str) for item in shape_ids)
        or shape_ids != sorted(shape_ids)
        or len(shape_ids) != len(set(shape_ids))
        or document.get("count") != len(shape_ids)
    ):
        raise ValueError("A train split ordering/count is invalid")
    test_path = path.with_name("test.json")
    test_document = json.loads(test_path.read_text(encoding="utf-8"))
    if (
        not isinstance(test_document, dict)
        or test_document.get("format") != "pcdenoise_shape_ids_v1"
        or test_document.get("format_version") != 1
        or test_document.get("split") != "test"
        or not isinstance(test_document.get("shape_ids"), list)
        or not isinstance(test_document.get("archive_sha256"), str)
    ):
        raise ValueError(f"invalid A test split: {test_path}")
    return list(shape_ids), test_document


def _read_b_lists(paths: list[Path]) -> list[tuple[str, str]]:
    mapped: list[tuple[str, str]] = []
    seen_source: set[str] = set()
    seen_target: set[str] = set()
    for path in paths:
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            source_id = raw_line.strip()
            if not source_id:
                continue
            match = _B_LINE_RE.fullmatch(source_id)
            if match is None:
                raise ValueError(
                    f"invalid B shape ID at {path}:{line_number}: "
                    f"{source_id!r}"
                )
            source_relative = source_id.removeprefix("shapenet/")
            target_id = f"00000000/b0000000{match.group(1)}"
            if source_relative in seen_source or target_id in seen_target:
                raise ValueError(f"duplicate B shape: {source_id}")
            seen_source.add(source_relative)
            seen_target.add(target_id)
            mapped.append((source_relative, target_id))
    if not mapped:
        raise ValueError("B train and validation lists are empty")
    return sorted(mapped, key=lambda item: item[1])


def _mesh_path(root: Path, shape_id: str) -> Path:
    return root / shape_id / "models" / "model_normalized.obj"


def _link_mesh(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"mesh must be a regular non-symlink file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=False)
    os.link(source, destination)


def main() -> int:
    args = _parser().parse_args()
    if args.output_mesh_root.exists():
        raise FileExistsError(args.output_mesh_root)
    if args.output_split_dir.exists():
        raise FileExistsError(args.output_split_dir)

    a_ids, a_test = _read_a_bundle(args.a_train_split)
    b_mapping = _read_b_lists(
        [args.b_train_list, args.b_validation_list]
    )
    b_ids = [target for _, target in b_mapping]
    all_ids = sorted(a_ids + b_ids)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("A and B mapped IDs overlap")

    source_manifest = build_stratified_split(
        all_ids, val_ratio=0.05, seed=args.seed
    )
    all_train_manifest = build_all_train_split(source_manifest)
    split_paths = write_split_artifacts(
        all_train_manifest,
        args.output_split_dir,
        test_shape_ids=a_test["shape_ids"],
        test_archive_sha256=a_test["archive_sha256"],
    )

    stage = _create_owned_stage(args.output_mesh_root)
    try:
        for shape_id in a_ids:
            _link_mesh(
                _mesh_path(args.a_mesh_root, shape_id),
                _mesh_path(stage.path, shape_id),
            )
        for source_id, target_id in b_mapping:
            _link_mesh(
                _mesh_path(args.b_mesh_root, source_id),
                _mesh_path(stage.path, target_id),
            )
        mapping_manifest = {
            "format": "pcdenoise_ab_mesh_index_v1",
            "format_version": 1,
            "mapping": "btrain_<24hex> -> b0000000<24hex>",
            "link_type": "hardlink",
            "counts": {
                "a": len(a_ids),
                "b_train_plus_validation": len(b_ids),
                "total": len(all_ids),
            },
            "seed": args.seed,
            "source_sha256": {
                "a_train_json": _sha256(args.a_train_split),
                "b_train_list": _sha256(args.b_train_list),
                "b_validation_list": _sha256(args.b_validation_list),
            },
            "combined_split_sha256": all_train_manifest["split_sha256"],
            "b_id_map": [
                {"source": source, "target": target}
                for source, target in b_mapping
            ],
        }
        (stage.path / "_ab_manifest.json").write_text(
            json.dumps(mapping_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_mesh_root = _publish_owned_stage(stage)
    except BaseException:
        _cleanup_owned_stage(stage)
        raise

    report = {
        "status": "READY",
        "counts": mapping_manifest["counts"],
        "output_mesh_root": os.fspath(output_mesh_root.resolve()),
        "output_split_dir": os.fspath(args.output_split_dir.resolve()),
        "train_json_sha256": _sha256(split_paths["train_json"]),
        "split_manifest_sha256": _sha256(split_paths["manifest"]),
        "split_sha256": all_train_manifest["split_sha256"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

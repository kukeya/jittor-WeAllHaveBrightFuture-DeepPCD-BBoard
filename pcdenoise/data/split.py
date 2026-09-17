"""Deterministic, category-stratified train/validation splits."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import zipfile
from collections import defaultdict
from numbers import Integral
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

from pcdenoise.data.archive import (
    _cleanup_owned_stage,
    _create_owned_stage,
    _publish_owned_stage,
)


SHAPE_ID_RE = re.compile(r"^[0-9]{8}/[0-9a-f]{28,32}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STRATIFIED_SPLIT_ALGORITHM = "per_synset_sha256_seeded_mt19937"
ALL_TRAIN_SPLIT_ALGORITHM = "all_official_train_v1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validated_ids(shape_ids: Iterable[str]) -> List[str]:
    ids = list(shape_ids)
    if not ids:
        raise ValueError("shape_ids must not be empty")
    non_strings = [item for item in ids if not isinstance(item, str)]
    if non_strings:
        raise ValueError(
            f"shape IDs must be strings: {non_strings[:3]!r}"
        )
    invalid = sorted(item for item in ids if not SHAPE_ID_RE.fullmatch(item))
    if invalid:
        raise ValueError(f"invalid shape IDs: {invalid[:3]!r}")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate shape IDs are forbidden")
    return sorted(ids)


def build_stratified_split(
    shape_ids: Iterable[str],
    val_ratio: float = 0.05,
    seed: int = 20260726,
) -> Dict[str, object]:
    """Build a traversal-order-independent split with per-synset guarantees."""

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    actual_seed = int(seed)
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be strictly between zero and one")
    ids = _validated_ids(shape_ids)
    groups: Dict[str, List[str]] = defaultdict(list)
    for shape_id in ids:
        groups[shape_id.split("/", 1)[0]].append(shape_id)

    train: List[str] = []
    validation: List[str] = []
    per_synset: Dict[str, Dict[str, int]] = {}
    for synset in sorted(groups):
        group = sorted(groups[synset])
        if len(group) == 1:
            selected_validation = set()
        else:
            count = int(len(group) * val_ratio + 0.5)
            count = max(1, min(len(group) - 1, count))
            local_seed = int.from_bytes(
                hashlib.sha256(
                    f"{actual_seed}:{synset}".encode("ascii")
                ).digest()[:8],
                byteorder="big",
            )
            shuffled = list(group)
            random.Random(local_seed).shuffle(shuffled)
            selected_validation = set(shuffled[:count])
        group_train = [item for item in group if item not in selected_validation]
        group_validation = [
            item for item in group if item in selected_validation
        ]
        train.extend(group_train)
        validation.extend(group_validation)
        per_synset[synset] = {
            "total": len(group),
            "train": len(group_train),
            "validation": len(group_validation),
        }

    train.sort()
    validation.sort()
    input_sha256 = hashlib.sha256(
        ("\n".join(ids) + "\n").encode("ascii")
    ).hexdigest()
    manifest: Dict[str, object] = {
        "format_version": 1,
        "algorithm": STRATIFIED_SPLIT_ALGORITHM,
        "seed": actual_seed,
        "val_ratio": float(val_ratio),
        "singleton_policy": "train",
        "input_sha256": input_sha256,
        "counts": {
            "total": len(ids),
            "train": len(train),
            "validation": len(validation),
        },
        "synset_counts": per_synset,
        "train": train,
        "validation": validation,
    }
    manifest["split_sha256"] = _canonical_hash(manifest)
    return manifest


def _validated_stratified_split_manifest(
    manifest: Mapping[str, object],
) -> Dict[str, object]:
    try:
        raw_train = list(manifest["train"])
        raw_validation = list(manifest["validation"])
        _validated_ids(raw_train + raw_validation)
        train = sorted(raw_train)
        validation = sorted(raw_validation)
        expected = build_stratified_split(
            train + validation,
            val_ratio=manifest["val_ratio"],
            seed=manifest["seed"],
        )
        supplied_snapshot = dict(manifest)
        supplied_bytes = _canonical_bytes(supplied_snapshot)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "manifest must be self-consistent builder output"
        ) from error
    if supplied_bytes != _canonical_bytes(expected):
        raise ValueError("manifest must be self-consistent builder output")
    return expected


def _all_train_manifest(
    shape_ids: Iterable[str],
    *,
    source_split_sha256: str,
) -> Dict[str, object]:
    ids = _validated_ids(shape_ids)
    if (
        not isinstance(source_split_sha256, str)
        or not SHA256_RE.fullmatch(source_split_sha256)
    ):
        raise ValueError("source_split_sha256 must be a lowercase SHA256")
    synset_counts = {
        synset: {
            "total": count,
            "train": count,
            "validation": 0,
        }
        for synset, count in _counts_by_synset(ids).items()
    }
    input_sha256 = hashlib.sha256(
        ("\n".join(ids) + "\n").encode("ascii")
    ).hexdigest()
    manifest: Dict[str, object] = {
        "format_version": 1,
        "algorithm": ALL_TRAIN_SPLIT_ALGORITHM,
        "source_split_sha256": source_split_sha256,
        "input_sha256": input_sha256,
        "counts": {
            "total": len(ids),
            "train": len(ids),
            "validation": 0,
        },
        "synset_counts": synset_counts,
        "train": ids,
        "validation": [],
    }
    manifest["split_sha256"] = _canonical_hash(manifest)
    return manifest


def build_all_train_split(
    source_manifest: Mapping[str, object],
) -> Dict[str, object]:
    """Merge a valid stratified split into one provenance-bound train split."""

    try:
        validated_source = _validated_split_manifest(source_manifest)
    except ValueError as error:
        raise ValueError(
            "source split must be valid stratified builder output"
        ) from error
    if validated_source.get("algorithm") != STRATIFIED_SPLIT_ALGORITHM:
        raise ValueError(
            "source split must be valid stratified builder output"
        )
    source_ids = list(validated_source["train"]) + list(
        validated_source["validation"]
    )
    try:
        manifest = _all_train_manifest(
            source_ids,
            source_split_sha256=str(validated_source["split_sha256"]),
        )
    except ValueError as error:
        raise ValueError(
            "source split must contain one complete unique ID universe"
        ) from error
    if manifest["input_sha256"] != validated_source["input_sha256"]:
        raise ValueError("source split input hash does not match its ID universe")
    return manifest


def _validated_all_train_split_manifest(
    manifest: Mapping[str, object],
) -> Dict[str, object]:
    try:
        raw_train = list(manifest["train"])
        raw_validation = list(manifest["validation"])
        if raw_validation:
            raise ValueError("all-train validation must be empty")
        expected = _all_train_manifest(
            raw_train,
            source_split_sha256=manifest["source_split_sha256"],
        )
        supplied_snapshot = dict(manifest)
        supplied_bytes = _canonical_bytes(supplied_snapshot)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "manifest must be self-consistent builder output"
        ) from error
    if supplied_bytes != _canonical_bytes(expected):
        raise ValueError("manifest must be self-consistent builder output")
    return expected


def _validated_split_manifest(
    manifest: Mapping[str, object],
) -> Dict[str, object]:
    """Return exact builder output or reject stale/inconsistent provenance."""

    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a self-consistent mapping")
    algorithm = manifest.get("algorithm")
    if algorithm == STRATIFIED_SPLIT_ALGORITHM:
        return _validated_stratified_split_manifest(manifest)
    if algorithm == ALL_TRAIN_SPLIT_ALGORITHM:
        return _validated_all_train_split_manifest(manifest)
    raise ValueError("manifest must be self-consistent builder output")


def write_split_artifacts(
    manifest: Mapping[str, object],
    output_dir: os.PathLike[str] | str,
    *,
    test_shape_ids: Iterable[str],
    test_archive_sha256: str,
    before_publish: Callable[[Path, int, int, bool], None] | None = None,
) -> Dict[str, Path]:
    """Atomically publish structured train/validation/test ID lists."""

    validated_manifest = _validated_split_manifest(manifest)
    train = list(validated_manifest["train"])
    validation = list(validated_manifest["validation"])
    split_sha256 = str(validated_manifest["split_sha256"])
    test = _validated_ids(test_shape_ids)
    if (
        not isinstance(test_archive_sha256, str)
        or not SHA256_RE.fullmatch(test_archive_sha256)
    ):
        raise ValueError("test_archive_sha256 must be a lowercase SHA256")

    stage = _create_owned_stage(output_dir)
    train_path = stage.path / "train.txt"
    validation_path = stage.path / "validation.txt"
    manifest_path = stage.path / "manifest.json"
    train_json_path = stage.path / "train.json"
    validation_json_path = stage.path / "validation.json"
    test_json_path = stage.path / "test.json"

    def structured_list(
        split_name: str,
        shape_ids: List[str],
        *,
        source_hash_name: str,
        source_hash: str,
    ) -> Dict[str, object]:
        return {
            "format": "pcdenoise_shape_ids_v1",
            "format_version": 1,
            "split": split_name,
            "count": len(shape_ids),
            "shape_ids": shape_ids,
            source_hash_name: source_hash,
        }

    try:
        train_path.write_text(
            "".join(f"{item}\n" for item in train), encoding="utf-8"
        )
        validation_path.write_text(
            "".join(f"{item}\n" for item in validation), encoding="utf-8"
        )
        manifest_path.write_text(
            json.dumps(validated_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        train_json_path.write_text(
            json.dumps(
                structured_list(
                    "train",
                    train,
                    source_hash_name="split_sha256",
                    source_hash=split_sha256,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        validation_json_path.write_text(
            json.dumps(
                structured_list(
                    "validation",
                    validation,
                    source_hash_name="split_sha256",
                    source_hash=split_sha256,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        test_json_path.write_text(
            json.dumps(
                structured_list(
                    "test",
                    test,
                    source_hash_name="archive_sha256",
                    source_hash=test_archive_sha256,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if before_publish is not None:
            before_publish(
                stage.destination,
                stage.device,
                stage.inode,
                True,
            )
        output = _publish_owned_stage(stage)
    except BaseException:
        _cleanup_owned_stage(stage)
        raise
    return {
        "manifest": output / "manifest.json",
        "train": output / "train.txt",
        "validation": output / "validation.txt",
        "train_json": output / "train.json",
        "validation_json": output / "validation.json",
        "test_json": output / "test.json",
    }


def _parse_official_lines(payload: bytes, source: str) -> List[str]:
    ids = []
    for line_number, raw_line in enumerate(
        payload.decode("utf-8").splitlines(), start=1
    ):
        value = raw_line.strip()
        if not value:
            continue
        if value.startswith("shapenet/"):
            value = value[len("shapenet/") :]
        if not SHAPE_ID_RE.fullmatch(value):
            raise ValueError(
                f"invalid official ID in {source}:{line_number}: {value!r}"
            )
        ids.append(value)
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate IDs in {source}")
    return ids


def load_official_split_from_starter(
    starter_zip: os.PathLike[str] | str,
    known_shape_ids: Sequence[str] | None = None,
) -> Dict[str, object]:
    """Read the Starter split solely as an audit reference.

    This function never feeds the biased official validation list into
    :func:`build_stratified_split`.
    """

    suffixes = {
        "train": "/datalist/train.txt",
        "validation": "/datalist/validate.txt",
    }
    with zipfile.ZipFile(starter_zip, mode="r") as archive:
        selected: Dict[str, str] = {}
        names = archive.namelist()
        for split_name, suffix in suffixes.items():
            matches = [name for name in names if name.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one {suffix!r}, found {len(matches)}"
                )
            selected[split_name] = matches[0]
        train = _parse_official_lines(
            archive.read(selected["train"]), selected["train"]
        )
        validation = _parse_official_lines(
            archive.read(selected["validation"]), selected["validation"]
        )
    overlap = sorted(set(train) & set(validation))
    if overlap:
        raise ValueError(f"official train/validation overlap: {overlap[:3]!r}")
    known = set(known_shape_ids or ())
    combined = set(train) | set(validation)
    unknown = sorted(combined - known) if known_shape_ids is not None else []
    missing = sorted(known - combined) if known_shape_ids is not None else []
    return {
        "format_version": 1,
        "purpose": "audit_only",
        "source_archive": str(Path(starter_zip).resolve()),
        "counts": {"train": len(train), "validation": len(validation)},
        "train": train,
        "validation": validation,
        "unknown_ids": unknown,
        "known_ids_missing_from_official": missing,
        "train_synset_counts": _counts_by_synset(train),
        "validation_synset_counts": _counts_by_synset(validation),
        "split_sha256": _canonical_hash(
            {"train": train, "validation": validation}
        ),
    }


def _counts_by_synset(shape_ids: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for shape_id in shape_ids:
        synset = shape_id.split("/", 1)[0]
        counts[synset] = counts.get(synset, 0) + 1
    return dict(sorted(counts.items()))

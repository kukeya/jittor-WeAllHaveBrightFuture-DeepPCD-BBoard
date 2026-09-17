"""Build the PGD2 train-ID contract with validation IDs removed."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

from .split import SHA256_RE, SHAPE_ID_RE


PGD2_TRAIN_SPLIT_FORMAT = "pcdenoise_pgd2_train_split_v1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids(values: Iterable[object], *, name: str) -> list[str]:
    result = list(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(
        not isinstance(value, str) or SHAPE_ID_RE.fullmatch(value) is None
        for value in result
    ):
        raise ValueError(f"{name} contains an invalid shape ID")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate shape IDs")
    return sorted(result)


def _read_source_split(path: Path) -> list[str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid source train split: {path}") from error
    if not isinstance(document, Mapping):
        raise ValueError("source train split root must be a mapping")
    if document.get("format") != "pcdenoise_shape_ids_v1":
        raise ValueError("source train split format is invalid")
    raw_ids = document.get("shape_ids")
    if not isinstance(raw_ids, list):
        raise ValueError("source train split shape_ids must be a list")
    result = _ids(raw_ids, name="source train split")
    if document.get("count") != len(result) or raw_ids != result:
        raise ValueError("source train split count/order is invalid")
    return result


def _read_exclusions(path: Path) -> tuple[list[str], str]:
    try:
        payload = path.read_bytes()
        lines = payload.decode("utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"invalid exclusion list: {path}") from error
    values = [line.strip() for line in lines if line.strip()]
    return (
        _ids(values, name="exclusion list"),
        hashlib.sha256(payload).hexdigest(),
    )


def _expected_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def build_pgd2_train_split(
    *,
    source_train_split: Path | str,
    exclude_sample_ids: Path | str,
    output_dir: Path | str,
    expected_exclusion_file_sha256: str,
    expected_retained_train_split_sha256: str,
) -> dict[str, object]:
    """Publish one immutable split directory for PGD2 cache construction."""

    source = Path(source_train_split).resolve()
    exclusions_source = Path(exclude_sample_ids).resolve()
    destination = Path(output_dir).resolve()
    expected_exclusion_sha256 = _expected_sha256(
        expected_exclusion_file_sha256,
        name="expected exclusion file SHA256",
    )
    expected_retained_sha256 = _expected_sha256(
        expected_retained_train_split_sha256,
        name="expected retained train split SHA256",
    )
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")

    source_ids = _read_source_split(source)
    exclusions, exclusion_file_sha256 = _read_exclusions(exclusions_source)
    if exclusion_file_sha256 != expected_exclusion_sha256:
        raise ValueError("exclusion file SHA256 mismatch")
    source_set = set(source_ids)
    overlap = sorted(source_set.intersection(exclusions))
    retained = sorted(source_set.difference(overlap))
    if not overlap:
        raise ValueError("exclusion list has no overlap with source train split")
    if not retained:
        raise ValueError("exclusions remove the complete train split")

    train_document: dict[str, object] = {
        "format": "pcdenoise_shape_ids_v1",
        "format_version": 1,
        "split": "train",
        "count": len(retained),
        "shape_ids": retained,
    }
    train_document["split_sha256"] = hashlib.sha256(
        _canonical_bytes(train_document)
    ).hexdigest()
    if train_document["split_sha256"] != expected_retained_sha256:
        raise ValueError("retained train split SHA256 mismatch")
    manifest: dict[str, object] = {
        "format": PGD2_TRAIN_SPLIT_FORMAT,
        "format_version": 1,
        "status": "completed",
        "source_count": len(source_ids),
        "excluded_count": len(overlap),
        "retained_count": len(retained),
        "excluded_ids": overlap,
        "retained_ids_sha256": hashlib.sha256(
            ("\n".join(retained) + "\n").encode("ascii")
        ).hexdigest(),
        "train_split_sha256": train_document["split_sha256"],
        "sources": {
            "source_train_split": os.fspath(source),
            "source_train_split_file_sha256": _sha256_file(source),
            "exclude_sample_ids": os.fspath(exclusions_source),
            "exclude_sample_ids_file_sha256": exclusion_file_sha256,
        },
    }
    manifest["content_sha256"] = hashlib.sha256(
        _canonical_bytes(manifest)
    ).hexdigest()

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.stage-", dir=destination.parent
        )
    )
    try:
        payloads = {
            "train.json": json.dumps(
                train_document,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
            "sample_ids.txt": "\n".join(retained) + "\n",
            "manifest.json": json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
        }
        for filename, payload in payloads.items():
            path = stage / filename
            with path.open("x", encoding="ascii") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        os.rename(stage, destination)
    except BaseException:
        for child in stage.iterdir():
            child.unlink()
        stage.rmdir()
        raise
    return manifest


__all__ = ["PGD2_TRAIN_SPLIT_FORMAT", "build_pgd2_train_split"]

#!/usr/bin/env python3
"""Create a relocatable PGD2 training config from generated local assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--paired-cache", type=Path, required=True)
    validation = parser.add_argument_group("optional historical validation binding")
    validation.add_argument("--validation-root", type=Path)
    validation.add_argument("--pgd1-validation-root", type=Path)
    validation.add_argument("--sample-ids", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.template.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("template root must be a mapping")

    paired_root = args.paired_cache.resolve()
    paired_manifest = _json(paired_root / "manifest.json")
    if (
        paired_manifest.get("status") != "completed"
        or paired_manifest.get("format") != "pcdenoise_pgd2_paired_cache_v1"
    ):
        raise ValueError("paired cache manifest is not completed")

    data = config.get("data")
    evaluation = config.get("evaluation")
    if not isinstance(data, dict) or not isinstance(evaluation, dict):
        raise ValueError("template must contain data and evaluation mappings")
    data["paired_cache"] = str(paired_root)
    data["expected_content_sha256"] = paired_manifest["content_sha256"]

    validation_values = (
        args.validation_root,
        args.pgd1_validation_root,
        args.sample_ids,
    )
    if any(value is not None for value in validation_values):
        if not all(value is not None for value in validation_values):
            raise ValueError(
                "validation-root, pgd1-validation-root, and sample-ids "
                "must be supplied together"
            )
        validation_root = args.validation_root.resolve()
        pgd1_validation_root = args.pgd1_validation_root.resolve()
        sample_ids = args.sample_ids.resolve()
        pgd1_manifest_path = pgd1_validation_root / "inference_manifest.json"
        pgd1_manifest = _json(pgd1_manifest_path)
        if (
            pgd1_manifest.get("status") != "completed"
            or pgd1_manifest.get("format") != "pcdenoise_prediction_v1"
        ):
            raise ValueError("PGD1 validation manifest is not completed")
        if not validation_root.is_dir() or not sample_ids.is_file():
            raise FileNotFoundError("validation root or sample ID file is missing")
        evaluation["pgd1_prediction_root"] = str(pgd1_validation_root)
        evaluation["authoritative_noisy_root"] = str(validation_root)
        evaluation["sample_ids"] = str(sample_ids)
        evaluation["expected_pgd1_manifest_sha256"] = _sha256(
            pgd1_manifest_path
        )
        evaluation["expected_sample_ids_sha256"] = _sha256(sample_ids)
    else:
        config["evaluation"] = {"enabled": False}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

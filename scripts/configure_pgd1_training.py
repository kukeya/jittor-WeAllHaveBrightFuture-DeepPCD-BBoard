#!/usr/bin/env python3
"""Create a relocatable PGD1 config from a generated surface cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--mesh-root", type=Path, required=True)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
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
    if not isinstance(config, dict) or not isinstance(config.get("data"), dict):
        raise ValueError("template must contain a data mapping")

    mesh_root = args.mesh_root.resolve()
    train_split = args.train_split.resolve()
    train_cache = args.train_cache.resolve()
    split_manifest_path = train_split.with_name("manifest.json")
    cache_manifest = _json(train_cache / "manifest.json")
    if (
        cache_manifest.get("format") != "pcdenoise_train_surface_cache_v1"
        or cache_manifest.get("format_version") != 1
        or cache_manifest.get("shape_count") != 35_632
        or cache_manifest.get("num_points") != 50_000
    ):
        raise ValueError("surface cache manifest is not the A+B PGD1 cache")
    if not mesh_root.is_dir() or not train_split.is_file():
        raise FileNotFoundError("mesh root or train split is missing")
    if not split_manifest_path.is_file():
        raise FileNotFoundError(split_manifest_path)
    if _sha256(train_split) != cache_manifest.get("train_split_sha256"):
        raise ValueError("train split differs from the surface cache")
    if _sha256(split_manifest_path) != cache_manifest.get(
        "split_manifest_sha256"
    ):
        raise ValueError("split manifest differs from the surface cache")
    content_sha256 = cache_manifest.get("content_sha256")
    if not isinstance(content_sha256, str) or len(content_sha256) != 64:
        raise ValueError("surface cache content SHA256 is invalid")

    data = config["data"]
    data["mesh_root"] = str(mesh_root)
    data["train_split"] = str(train_split)
    data["train_cache"] = str(train_cache)
    data["expected_train_split_sha256"] = cache_manifest[
        "train_split_sha256"
    ]
    data["expected_split_manifest_sha256"] = cache_manifest[
        "split_manifest_sha256"
    ]
    data["expected_content_sha256"] = content_sha256

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

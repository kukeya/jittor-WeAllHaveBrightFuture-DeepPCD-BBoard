#!/usr/bin/env python3
"""Build a PGD2 train split after removing pinned validation IDs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdenoise.data.pgd2_split import build_pgd2_train_split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-train-split", type=Path, required=True)
    parser.add_argument("--exclude-sample-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-exclusion-file-sha256", required=True)
    parser.add_argument("--expected-retained-train-split-sha256", required=True)
    arguments = parser.parse_args()
    result = build_pgd2_train_split(
        source_train_split=arguments.source_train_split,
        exclude_sample_ids=arguments.exclude_sample_ids,
        output_dir=arguments.output_dir,
        expected_exclusion_file_sha256=(
            arguments.expected_exclusion_file_sha256
        ),
        expected_retained_train_split_sha256=(
            arguments.expected_retained_train_split_sha256
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

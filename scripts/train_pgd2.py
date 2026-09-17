#!/usr/bin/env python3
"""Train README PGD2 from one frozen PGD1-output/clean paired cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument("--initial-checkpoint", type=Path)
    start.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--expected-initial-checkpoint-sha256")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help=(
            "skip the detached per-epoch validation pass; this does not change "
            "training batches, losses, optimizer updates, or learning-rate schedule"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pcdenoise.training.pgd2_runner import load_pgd2_config, run_pgd2_training

    config, raw_config_sha256 = load_pgd2_config(
        args.config,
        require_evaluation=not args.skip_validation,
    )
    result = run_pgd2_training(
        config,
        raw_config_sha256=raw_config_sha256,
        run_dir=args.run_dir,
        initial_checkpoint=args.initial_checkpoint,
        expected_initial_checkpoint_sha256=(
            args.expected_initial_checkpoint_sha256
        ),
        resume_checkpoint=args.resume_checkpoint,
        run_validation=not args.skip_validation,
        command=sys.argv,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

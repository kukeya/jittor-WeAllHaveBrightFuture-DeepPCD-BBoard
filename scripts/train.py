#!/usr/bin/env python3
"""Train the pure-Jittor PGD denoiser from one immutable YAML config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", type=Path)
    checkpoint_group.add_argument(
        "--init-checkpoint",
        type=Path,
        help=(
            "load model/VQ weights but start a fresh optimizer and training "
            "schedule; intended for short fine-tuning with a new config"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pcdenoise.training.runner import run_training

    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid training config: {args.config}") from error
    if not isinstance(config, dict):
        raise ValueError("training config root must be a mapping")
    result = run_training(
        config,
        run_dir=args.run_dir,
        resume_checkpoint=args.resume,
        initial_checkpoint=args.init_checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

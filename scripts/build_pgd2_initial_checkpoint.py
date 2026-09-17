#!/usr/bin/env python3
"""Publish one common step-0 model snapshot for matched PGD2 arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jittor as jt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pcdenoise.models.factory import build_pgd_model
from pcdenoise.training.checkpoint import save_checkpoint
from pcdenoise.training.pgd2_runner import (
    load_pgd2_config,
    pgd2_model_config_sha256,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="construct the common snapshot on CUDA (CPU is sufficient)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="load only model, data, and training sections of the config",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    config, raw_config_sha256 = load_pgd2_config(
        args.config,
        require_evaluation=not args.skip_validation,
    )
    if not args.output.parent.is_dir():
        raise FileNotFoundError(args.output.parent)
    jt.flags.use_cuda = int(args.use_cuda)
    jt.set_global_seed(int(config["training"]["seed"]))
    model = build_pgd_model(config["model"], config["data"])
    result = save_checkpoint(
        args.output,
        model=model,
        optimizer=None,
        step=0,
        config_sha256=pgd2_model_config_sha256(config),
        metrics={
            "initialization_seed": int(config["training"]["seed"]),
        },
        training_state={
            "purpose": "matched_pgd2_step0_initialization",
            "raw_builder_config_sha256": raw_config_sha256,
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

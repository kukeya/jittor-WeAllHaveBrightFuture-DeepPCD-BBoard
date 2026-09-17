#!/usr/bin/env python3
"""Denoise competition-style inputs with one verified Jittor checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# The submitted inference path uses Jittor's native CUDA convolution kernels.
# Set this before importing Jittor so CUDA 12.4 does not require a separate
# cuDNN installation.
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--patch-size", type=int)
    parser.add_argument("--seed-k", type=float, default=6.0)
    parser.add_argument("--patch-batch-size", type=int, default=5)
    parser.add_argument("--niters", type=int, default=1)
    parser.add_argument(
        "--normalization-mode",
        choices=("noisy_max", "identity", "robust_quantile"),
        default="noisy_max",
        help="per-sample noisy-cloud normalization (default: noisy_max)",
    )
    parser.add_argument(
        "--robust-quantile",
        type=float,
        help=(
            "required only for --normalization-mode robust_quantile; "
            "strictly in (0,1]"
        ),
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("hard_best",),
        default="hard_best",
        help="overlap displacement fusion used by the submitted model",
    )
    parser.add_argument(
        "--iteration-damping",
        type=float,
        default=1.0,
        help="residual multiplier for pass 2 onward, strictly in (0,1]",
    )
    parser.add_argument(
        "--sample-ids",
        type=Path,
        help="optional UTF-8 file with one <synset>/<model> ID per line",
    )
    parser.add_argument(
        "--noise-route-diagnostic-dir",
        type=Path,
        help=(
            "optional sibling output for input-only estimated-noise route "
            "diagnostics; requires --niters 1 and hard_best fusion"
        ),
    )
    return parser


def _sample_ids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise ValueError("--sample-ids file is empty")
    return lines


def main() -> int:
    args = _parser().parse_args()
    # Import Jittor only after parsing so ``--help`` works on machines without
    # an available CUDA device.
    os.environ.setdefault("conv_opt", "1")
    import jittor as jt

    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pcdenoise.inference_config import load_inference_config
    from pcdenoise.models.factory import build_pgd_model, pgd_model_architecture
    from pcdenoise.prediction import run_prediction
    from pcdenoise.training.checkpoint import load_checkpoint

    config = load_inference_config(args.inference_config)
    config_sha256 = str(config["canonical_config_sha256"])
    model_config = config["model"]
    patch_size = (
        int(args.patch_size)
        if args.patch_size is not None
        else int(model_config["patch_size"])
    )
    if patch_size != int(model_config["patch_size"]):
        raise ValueError(
            "--patch-size must match the checkpoint model patch_size"
        )

    use_cuda = bool(config["use_cuda"])
    if use_cuda and not jt.has_cuda:
        raise RuntimeError("checkpoint config requires CUDA")
    jt.flags.use_cuda = int(use_cuda)
    model = build_pgd_model(model_config, config["data"])
    checkpoint = load_checkpoint(
        args.checkpoint,
        model=model,
        expected_config_sha256=config_sha256,
    )
    if (
        checkpoint["checkpoint_sha256"] != config["checkpoint_sha256"]
        or checkpoint["step"] != config["checkpoint_step"]
    ):
        raise ValueError(
            "checkpoint digest or step differs from the inference config"
        )
    model.eval()
    result = run_prediction(
        model,
        input_root=args.input_dir,
        output_dir=args.output_dir,
        patch_size=patch_size,
        seed_k=args.seed_k,
        patch_batch_size=args.patch_batch_size,
        niters=args.niters,
        normalization_mode=args.normalization_mode,
        robust_quantile=args.robust_quantile,
        fusion_mode=args.fusion_mode,
        iteration_damping=args.iteration_damping,
        sample_ids=_sample_ids(args.sample_ids),
        noise_route_diagnostic_dir=args.noise_route_diagnostic_dir,
        model_reference={
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "checkpoint_step": checkpoint["step"],
            "config_sha256": config_sha256,
            "architecture": pgd_model_architecture(model_config),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

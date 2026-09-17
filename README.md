# PGD Cascade Point-Cloud Denoising

This repository contains the B-stage point-cloud denoising code submitted by
team "我们都有光明的未来" to Track 2 of the Sixth CG Graphics AI Challenge.
The method trains PGD1 first, then trains PGD2 on the frozen PGD1 predictions
and clean targets. Inference applies PGD1 followed by two PGD2 passes.

The submitted B-stage result was `Total=81.90`, `CD=70.71`, and `P2S=93.09`.
The two checkpoints required by the inference command are included in
`checkpoints/`. Datasets, generated caches, logs, and submission results are
not included.

## Environment

Use Python 3.10 and create the environment from the tracked file:

```bash
conda env create -f environment.yaml
conda activate jittor
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## Data preparation

Obtain the official A/B training data and B-test inputs from the competition
organizer. Keep them outside Git; the required local layout is described in
[data/README.md](data/README.md).

```bash
python scripts/prepare_ab_full_training.py \
  --a-mesh-root /path/to/a_train/shapenet \
  --a-train-split /path/to/a_train_split.json \
  --b-mesh-root /path/to/b_train/shapenet \
  --b-train-list /path/to/b_train.txt \
  --b-validation-list /path/to/b_validation.txt \
  --output-mesh-root data/prepared/shapenet \
  --output-split-dir data/splits/ab_full_seed20260726 \
  --seed 20260726
```

## Training

The training seed and model parameters are in `configs/train/`. Build the
surface cache and generate a local config with
`scripts/build_train_surface_cache.py` and
`scripts/configure_pgd1_training.py`, then train PGD1:

```bash
python scripts/train.py \
  --config /path/to/generated_pgd1.yaml \
  --run-dir outputs/pgd1
```

For PGD2, generate the frozen PGD1-output/clean paired cache with
`build_pgd2_noisy_cache.py`, `run_pgd1_pgd2_shards.py`, and
`build_pgd2_paired_cache.py`; generate its local config with
`configure_pgd2_training.py`, then run `scripts/train_pgd2.py`.

## Inference

The required PGD1-e200 and PGD2-e139 checkpoints are in `checkpoints/`; verify
them with `sha256sum -c checkpoints/SHA256SUMS`.

```bash
python scripts/infer_pgd1_pgd2_pgd2.py \
  --project-root . \
  --input-root /path/to/b_test \
  --output-root outputs/b_predictions \
  --work-root outputs/b_work \
  --pgd1-inference-config configs/inference/pgd1_e200.json \
  --pgd1-checkpoint checkpoints/pgd1_e200_step_00712800.pkl \
  --pgd2-inference-config configs/inference/pgd2_e139.json \
  --pgd2-checkpoint checkpoints/pgd2_e139_epoch_0139_step_00494006.pkl \
  --devices 0 \
  --random-seed 20260819
```

## Results

CD is Chamfer distance and P2S is point-to-surface distance; the competition
score weights the two equally. The reported online result uses the original
checkpoints and official data, so a newly trained model may differ.

## License

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for the license and third-party
references.

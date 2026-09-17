# Deep Learning-Based 3D Point Cloud Denoising

## Method Design and Implementation — Track B

This repository contains the Track B implementation submitted to the Sixth CG Graphics AI Challenge. It trains PGD1 on the combined A+B mesh set, trains PGD2 on frozen PGD1 predictions, and applies the cascade PGD1 -> PGD2 -> PGD2 at inference.

The submitted Track B result was **Total 81.90**, **CD 70.71**, and **P2S 93.09**. Reproducing the result requires the official data, the released checkpoints, and the inference settings below.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Results and Reproducibility](#results-and-reproducibility)
- [License and Acknowledgement](#license-and-acknowledgement)

## Requirements

| Component | Tested version |
| --- | --- |
| OS | Ubuntu 22.04 |
| GPU | NVIDIA RTX 4090 |
| CUDA | 12.4 |
| Python | 3.10 |
| Jittor | 1.3.10.0 |
| Compiler | g++ 10 |

Released inference checkpoints:

~~~text
checkpoints/pgd1_e200_step_00712800.pkl
checkpoints/pgd2_e139_epoch_0139_step_00494006.pkl
~~~

Verify them before use:

~~~bash
(cd checkpoints && sha256sum -c SHA256SUMS)
~~~

## Installation

Create the environment from the repository root. Jittor's cache and the temporary directory must be writable. Change CUDA_HOME if CUDA is installed elsewhere.

~~~bash
conda env create -f environment.yaml
conda activate jittor

export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export CUDA_HOME=/usr/local/cuda-12.4
export nvcc_path="$CUDA_HOME/bin/nvcc"
export cc_path="$(command -v g++)"
export cuda_archs=89
export JITTOR_HOME=/absolute/writable/path/jittor_cache
export TMPDIR=/absolute/writable/path/tmp
mkdir -p "$JITTOR_HOME" "$TMPDIR"

export REPO_ROOT="$PWD"
export WORK_ROOT=/absolute/writable/path/point_denoising_work
mkdir -p "$WORK_ROOT"
~~~

## Data Preparation

The official data are not redistributed here. Set these paths to the archives supplied by the competition organizer.

~~~bash
export TRAIN_DATASET_A=/path/to/dataset_train.tar.gz
export TEST_DATASET_A=/path/to/dataset_test_noisy.zip
export TRAIN_DATASET_B=/path/to/dataset_train_b.zip
export DATALIST_B=/path/to/datalist.zip
export TEST_DATASET_B=/path/to/dataset_test_noisy_b.zip
~~~

Prepare the combined A+B training set and the deterministic PGD1 clean-surface cache.

~~~bash
python scripts/prepare_data.py --train-archive "$TRAIN_DATASET_A" --test-archive "$TEST_DATASET_A" --output-dir "$WORK_ROOT/data/a_prepared" --split-dir "$WORK_ROOT/data/a_split" --seed 20260726 --val-ratio 0.05

python scripts/build_all_train_split.py --source-split "$WORK_ROOT/data/a_split" --output-dir "$WORK_ROOT/data/a_all15833"

unzip -q "$TRAIN_DATASET_B" -d "$WORK_ROOT/data/b_raw"
unzip -q "$DATALIST_B" -d "$WORK_ROOT/data/b_lists"

python scripts/prepare_ab_full_training.py --a-mesh-root "$WORK_ROOT/data/a_prepared/train/dataset_train/shapenet" --a-train-split "$WORK_ROOT/data/a_all15833/train.json" --b-mesh-root "$WORK_ROOT/data/b_raw/dataset_train/shapenet" --b-train-list "$WORK_ROOT/data/b_lists/datalist/train_b.txt" --b-validation-list "$WORK_ROOT/data/b_lists/datalist/validate_b.txt" --output-mesh-root "$WORK_ROOT/data/ab35632/shapenet" --output-split-dir "$WORK_ROOT/data/ab35632_split" --seed 20260726

python scripts/build_train_surface_cache.py --mesh-root "$WORK_ROOT/data/ab35632/shapenet" --train-split "$WORK_ROOT/data/ab35632_split/train.json" --output-dir "$WORK_ROOT/cache/surface50000_ab35632" --select-count 35632 --num-points 50000 --expected-train-split-sha256 a12ad101f5d4171e86e4684aa11630c8ebd99e5e84d45756e1bdce1c0d5a822b --expected-split-manifest-sha256 9403dfe2b71fa8292ad093c545beae7bc1fc0c28bd25d690ba70655309de05a0 --seed 20260726 --workers 16
~~~

The resulting cache contains 35,632 finite float32 arrays of shape (50000, 3). Keep the cache manifests unchanged.

## Training

### PGD1

Create a local configuration bound to the prepared cache, then train PGD1 with seed 20260726.

~~~bash
mkdir -p "$WORK_ROOT/configs"

python scripts/configure_pgd1_training.py --template configs/train/pgd_ab35632_gate_e200_noisyfit_cube24_seed20260726.yaml --mesh-root "$WORK_ROOT/data/ab35632/shapenet" --train-split "$WORK_ROOT/data/ab35632_split/train.json" --train-cache "$WORK_ROOT/cache/surface50000_ab35632" --output "$WORK_ROOT/configs/pgd1_e200.yaml"

PYTHONHASHSEED=20260726 CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --config "$WORK_ROOT/configs/pgd1_e200.yaml" --run-dir "$WORK_ROOT/runs/pgd1_e200"
~~~

### PGD2

PGD2 is trained on frozen PGD1 outputs and their clean targets. The commands below construct the paired cache and train PGD2 with seed 20260813.

~~~bash
export PGD1_CONFIG_SHA256=fba86f44277df8d81ff301e6201a4d1157e0a3052658f32f61e8bda572e22d99
export PGD1_CHECKPOINT_SHA256=d679e174564442f1f6c031b50fbd45f94d6035c7c089e555be5aa54f2834f2c0
export VAL200_IDS_SHA256=09816d7c0242bd53670a64ab60ddde70fa97a02b5e6bd5afeb44fd69cbda124b
export PGD2_SPLIT_SHA256=c532f6745c76a1f46a0897a613a24a1a4a18ba0e86f413468d93b4eac5c6b33c

python scripts/build_pgd2_train_split.py --source-train-split "$WORK_ROOT/data/ab35632_split/train.json" --exclude-sample-ids configs/train/pgd2_val200_excluded_ids.txt --output-dir "$WORK_ROOT/data/pgd2_train_35534" --expected-exclusion-file-sha256 "$VAL200_IDS_SHA256" --expected-retained-train-split-sha256 "$PGD2_SPLIT_SHA256"

python scripts/build_pgd2_noisy_cache.py --clean-cache "$WORK_ROOT/cache/surface50000_ab35632" --include-split "$WORK_ROOT/data/pgd2_train_35534/train.json" --output-dir "$WORK_ROOT/cache/pgd2_noisy_epoch1" --epoch 1 --seed 20260813 --scale-min 0.005 --scale-max 0.020 --expected-shape-count 35534 --workers 16

PYTHONHASHSEED=20260726 python scripts/run_pgd1_pgd2_shards.py --input-cache "$WORK_ROOT/cache/pgd2_noisy_epoch1" --output-root "$WORK_ROOT/cache/pgd1_e200_outputs" --inference-config configs/inference/pgd1_e200.json --checkpoint checkpoints/pgd1_e200_step_00712800.pkl --expected-sample-count 35534 --gpus 0 --patch-size 1000 --seed-k 6 --patch-batch-size 20 --fusion-mode hard_best --cuda-home "$CUDA_HOME" --cc-path "$cc_path" --jittor-home-root "$WORK_ROOT/jittor/pgd1_paired"

python scripts/build_pgd2_paired_cache.py --clean-cache "$WORK_ROOT/cache/surface50000_ab35632" --input-cache "$WORK_ROOT/cache/pgd2_noisy_epoch1" --shard-root "$WORK_ROOT/cache/pgd1_e200_outputs" --output-dir "$WORK_ROOT/cache/pgd2_paired_35534" --expected-pgd1-checkpoint-sha256 "$PGD1_CHECKPOINT_SHA256" --expected-pgd1-config-sha256 "$PGD1_CONFIG_SHA256"

python scripts/configure_pgd2_training.py --template configs/train/pgd2_e200pair_b40_e140_warmup_hold_cosine_seed20260813.yaml --paired-cache "$WORK_ROOT/cache/pgd2_paired_35534" --output "$WORK_ROOT/configs/pgd2_e140.yaml"

mkdir -p "$WORK_ROOT/checkpoints"
PYTHONHASHSEED=20260813 CUDA_VISIBLE_DEVICES=0 python scripts/build_pgd2_initial_checkpoint.py --config "$WORK_ROOT/configs/pgd2_e140.yaml" --output "$WORK_ROOT/checkpoints/pgd2_step0.pkl" --use-cuda --skip-validation

export PGD2_INITIAL_SHA256="$(sha256sum "$WORK_ROOT/checkpoints/pgd2_step0.pkl" | awk '{print $1}')"
PYTHONHASHSEED=20260813 CUDA_VISIBLE_DEVICES=0 python -u scripts/train_pgd2.py --config "$WORK_ROOT/configs/pgd2_e140.yaml" --run-dir "$WORK_ROOT/runs/pgd2_from_scratch" --initial-checkpoint "$WORK_ROOT/checkpoints/pgd2_step0.pkl" --expected-initial-checkpoint-sha256 "$PGD2_INITIAL_SHA256" --skip-validation
~~~

The released PGD2-e139 checkpoint is the selected historical checkpoint. Training omits the private validation geometry and therefore uses --skip-validation; the training batches, loss, optimizer, and schedule are unchanged.

## Inference

The following command runs the released cascade on the official Track B noisy-test archive without retraining. It writes the final denoised arrays and an inference manifest to $WORK_ROOT/pred/final.

~~~bash
mkdir -p "$WORK_ROOT/data/b_test"
unzip -q "$TEST_DATASET_B" -d "$WORK_ROOT/data/b_test"

PYTHONHASHSEED=20260819 CUDA_VISIBLE_DEVICES=0 python -u scripts/infer_pgd1_pgd2_pgd2.py --project-root "$REPO_ROOT" --input-root "$WORK_ROOT/data/b_test/dataset_test_noisy" --output-root "$WORK_ROOT/pred/final" --work-root "$WORK_ROOT/pred/cascade_work" --pgd1-inference-config configs/inference/pgd1_e200.json --pgd1-checkpoint checkpoints/pgd1_e200_step_00712800.pkl --pgd2-inference-config configs/inference/pgd2_e139.json --pgd2-checkpoint checkpoints/pgd2_e139_epoch_0139_step_00494006.pkl --devices 0 --pgd1-device 0 --seed-k 6 --alpha 1.12 --beta 0.59 --pgd1-patch-batch-size 20 --pgd2-patch-batch-size 160 --random-seed 20260819 --jittor-home-root "$WORK_ROOT/jittor/final_cascade"
~~~

Successful inference writes 200 finite float32 arrays of shape (50000, 3) and a completed $WORK_ROOT/pred/cascade_work/cascade_status.json.

## Results and Reproducibility

| Item | Submitted setting |
| --- | --- |
| PGD1 | e200, step 712800 |
| PGD2 | e139, step 494006 |
| Patch size | 1000 |
| Patch coverage | seed_k=6, hard_best |
| FPS plans | index0, centroid_far, x_min, x_max |
| PGD2 residual scales | alpha=1.12, then beta=0.59 |
| Inference seed | 20260819 |

CD is bidirectional Chamfer distance and P2S is point-to-surface distance. A local rerun can vary numerically across CUDA and Jittor environments; it is not an independent verification of the online leaderboard score.

## License and Acknowledgement

See [LICENSE](LICENSE) and [NOTICE](NOTICE). The method is informed by [Guiding Point Cloud Denoising with Learned Structural Priors](https://github.com/git-guocc/PGD).

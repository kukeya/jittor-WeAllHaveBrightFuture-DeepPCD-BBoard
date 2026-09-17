# Data layout

This directory contains documentation only. Do not commit official
competition data, meshes, point clouds, caches, predictions, or labels.

Keep the data in a local directory (or create a local symlink named `data`)
with this shape after running the preparation commands:

```text
data/
├── prepared/
│   └── shapenet/<synset>/<model_id>/models/model_normalized.obj
├── splits/
│   └── ab_full_seed20260726/
│       ├── train.json
│       └── manifest.json
├── b_test/
│   └── shapenet/<synset>/<model_id>/noisy.npy
└── validation/                 # optional authorized local validation set
```

`noisy.npy`, `clean.npy`, and `denoised.npy` must be finite `float32` arrays
with shape `(50000, 3)`. The B-test input normally contains 200 samples; the
official test labels must never be used by inference.

The competition data are not redistributable through this repository. Obtain
them from the organizer and configure paths through CLI arguments or generated
YAML files. Generated `outputs/` and local `data/` content are ignored by
Git.

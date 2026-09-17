"""README-faithful fixed-pair PGD2 training and validation.

This module is deliberately separate from :mod:`pcdenoise.training.runner`.
The latter builds online noisy-to-clean PGD1 batches, while PGD2 consumes one
frozen ``PGD1 output -> clean`` paired cache and must never regenerate noise.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import socket
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

import jittor as jt
import numpy as np
import yaml
from tensorboardX import SummaryWriter

from pcdenoise.data.pgd2_training import (
    PGD2EpochPlan,
    PGD2PairedDataset,
    build_pgd2_epoch_plan,
    build_pgd2_training_batch,
    load_pgd2_paired_dataset,
    pgd2_epoch_batches,
)
from pcdenoise.models.factory import build_pgd_model, pgd_model_architecture
from pcdenoise.models.losses import (
    INFOCD_PROFILES,
    correspondence_huber_loss,
    infocd_loss,
)
from pcdenoise.prediction import run_prediction
from pcdenoise.training.checkpoint import load_checkpoint, save_checkpoint
from pcdenoise.training.engine import canonical_config_sha256
from pcdenoise.training.pgd2_schedule import (
    canonicalize_pgd2_schedule_config,
    pgd2_learning_rate_at_step,
)


PGD2_CONFIG_FORMAT = "pcdenoise_pgd2_train_v1"
PGD2_SAMPLER = "pgd2_epoch_four_patch_v1"
_SHA256_CHARS = frozenset("0123456789abcdef")
_EPOCH_CHECKPOINT_PATTERN = re.compile(
    r"epoch_([0-9]{4})_step_([0-9]{8})[.]pkl"
)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _keys(
    value: Mapping[str, object],
    *,
    name: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    accepted = required | (optional or set())
    missing = sorted(required - set(value))
    extra = sorted(set(value) - accepted)
    if missing or extra:
        raise ValueError(f"{name} fields are invalid: missing={missing}, extra={extra}")


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _real(
    value: object,
    *,
    name: str,
    minimum: float = 0.0,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (positive and result <= minimum):
        relation = ">" if positive else ">="
        raise ValueError(f"{name} must be finite and {relation} {minimum}")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_CHARS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _resolved_existing_path(value: object, *, name: str, directory: bool) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty path string")
    path = Path(value).expanduser().resolve()
    if directory and not path.is_dir():
        raise NotADirectoryError(path)
    if not directory and not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def _canonical_model_config(value: object) -> dict[str, object]:
    source = _mapping(value, name="model")
    _keys(
        source,
        name="model",
        required={
            "architecture",
            "patch_size",
            "feature_dims",
            "codebook_sizes",
            "temperature",
            "ema_momentum",
            "displacement_scale",
        },
        optional={"noise_conditioning", "pointwise_displacement_gate"},
    )
    if source["architecture"] != "pgd":
        raise ValueError("PGD2 baseline requires model.architecture='pgd'")
    patch_size = _integer(source["patch_size"], name="model.patch_size", minimum=1)
    feature_dims = source["feature_dims"]
    codebook_sizes = source["codebook_sizes"]
    if not isinstance(feature_dims, (list, tuple)) or len(feature_dims) != 5:
        raise ValueError("model.feature_dims must contain five integers")
    if not isinstance(codebook_sizes, (list, tuple)) or len(codebook_sizes) != 4:
        raise ValueError("model.codebook_sizes must contain four integers")
    canonical = {
        "architecture": "pgd",
        "patch_size": patch_size,
        "feature_dims": [
            _integer(item, name="model.feature_dims item", minimum=1)
            for item in feature_dims
        ],
        "codebook_sizes": [
            _integer(item, name="model.codebook_sizes item", minimum=1)
            for item in codebook_sizes
        ],
        "temperature": _real(
            source["temperature"], name="model.temperature", positive=True
        ),
        "ema_momentum": _real(
            source["ema_momentum"], name="model.ema_momentum", minimum=0.0
        ),
        "displacement_scale": _real(
            source["displacement_scale"],
            name="model.displacement_scale",
            positive=True,
        ),
    }
    if not 0.0 <= canonical["ema_momentum"] < 1.0:
        raise ValueError("model.ema_momentum must lie in [0, 1)")
    for key in ("noise_conditioning", "pointwise_displacement_gate"):
        enabled = _boolean(source.get(key, False), name=f"model.{key}")
        if enabled:
            raise ValueError(
                f"README PGD2 baseline requires model.{key}=false; "
                "conditioning/gating would change more than the LR schedule"
            )
        canonical[key] = False
    return canonical


def validate_pgd2_training_config(
    value: object,
    *,
    require_evaluation: bool = True,
) -> dict[str, object]:
    """Return one strict, path-resolved PGD2 experiment configuration."""

    source = _mapping(value, name="config")
    _keys(
        source,
        name="config",
        required={"format", "model", "data", "training", "evaluation"},
    )
    if source["format"] != PGD2_CONFIG_FORMAT:
        raise ValueError(f"format must be {PGD2_CONFIG_FORMAT!r}")
    model = _canonical_model_config(source["model"])

    data_source = _mapping(source["data"], name="data")
    _keys(
        data_source,
        name="data",
        required={
            "paired_cache",
            "expected_content_sha256",
            "expected_pgd1_checkpoint_sha256",
            "expected_pgd1_config_sha256",
            "expected_sample_count",
            "expected_point_count",
        },
    )
    data = {
        "paired_cache": _resolved_existing_path(
            data_source["paired_cache"], name="data.paired_cache", directory=True
        ),
        "expected_content_sha256": _sha256(
            data_source["expected_content_sha256"],
            name="data.expected_content_sha256",
        ),
        "expected_pgd1_checkpoint_sha256": _sha256(
            data_source["expected_pgd1_checkpoint_sha256"],
            name="data.expected_pgd1_checkpoint_sha256",
        ),
        "expected_pgd1_config_sha256": _sha256(
            data_source["expected_pgd1_config_sha256"],
            name="data.expected_pgd1_config_sha256",
        ),
        "expected_sample_count": _integer(
            data_source["expected_sample_count"],
            name="data.expected_sample_count",
            minimum=1,
        ),
        "expected_point_count": _integer(
            data_source["expected_point_count"],
            name="data.expected_point_count",
            minimum=1,
        ),
    }

    training_source = _mapping(source["training"], name="training")
    _keys(
        training_source,
        name="training",
        required={
            "seed",
            "sampler",
            "max_epochs",
            "batch_size",
            "patches_per_shape",
            "prefetch_batches",
            "learning_rate",
            "learning_rate_schedule",
            "loss_profile",
            "reconstruction_weight",
            "point_mse_weight",
            "correspondence_huber_weight",
            "correspondence_huber_delta",
            "log_every",
            "use_cuda",
        },
    )
    if training_source["sampler"] != PGD2_SAMPLER:
        raise ValueError(f"training.sampler must be {PGD2_SAMPLER!r}")
    if training_source["loss_profile"] not in INFOCD_PROFILES:
        raise ValueError(f"training.loss_profile must be one of {INFOCD_PROFILES}")
    schedule = canonicalize_pgd2_schedule_config(
        {
            "learning_rate": training_source["learning_rate"],
            "max_epochs": training_source["max_epochs"],
            "learning_rate_schedule": training_source["learning_rate_schedule"],
        }
    )
    training = {
        "seed": _integer(training_source["seed"], name="training.seed"),
        "sampler": PGD2_SAMPLER,
        "max_epochs": schedule["max_epochs"],
        "batch_size": _integer(
            training_source["batch_size"], name="training.batch_size", minimum=1
        ),
        "patches_per_shape": _integer(
            training_source["patches_per_shape"],
            name="training.patches_per_shape",
            minimum=1,
        ),
        "prefetch_batches": _integer(
            training_source["prefetch_batches"],
            name="training.prefetch_batches",
            minimum=0,
        ),
        "learning_rate": schedule["learning_rate"],
        "learning_rate_schedule": schedule["learning_rate_schedule"],
        "loss_profile": str(training_source["loss_profile"]),
        "reconstruction_weight": _real(
            training_source["reconstruction_weight"],
            name="training.reconstruction_weight",
            minimum=0.0,
        ),
        "point_mse_weight": _real(
            training_source["point_mse_weight"],
            name="training.point_mse_weight",
            minimum=0.0,
        ),
        "correspondence_huber_weight": _real(
            training_source["correspondence_huber_weight"],
            name="training.correspondence_huber_weight",
            minimum=0.0,
        ),
        "correspondence_huber_delta": _real(
            training_source["correspondence_huber_delta"],
            name="training.correspondence_huber_delta",
            positive=True,
        ),
        "log_every": _integer(
            training_source["log_every"], name="training.log_every", minimum=1
        ),
        "use_cuda": _boolean(
            training_source["use_cuda"], name="training.use_cuda"
        ),
    }
    if training["patches_per_shape"] != 4:
        raise ValueError("README PGD2 experiment requires patches_per_shape=4")

    if not require_evaluation:
        return {
            "format": PGD2_CONFIG_FORMAT,
            "model": model,
            "data": data,
            "training": training,
            "evaluation": {"enabled": False},
        }

    evaluation_source = _mapping(source["evaluation"], name="evaluation")
    _keys(
        evaluation_source,
        name="evaluation",
        required={
            "pgd1_prediction_root",
            "authoritative_noisy_root",
            "sample_ids",
            "expected_pgd1_checkpoint_sha256",
            "expected_pgd1_config_sha256",
            "expected_pgd1_manifest_sha256",
            "expected_sample_ids_sha256",
            "expected_authoritative_inventory_sha256",
            "expected_sample_count",
            "patch_size",
            "seed_k",
            "patch_batch_size",
            "niters",
            "normalization_mode",
            "fusion_mode",
            "iteration_damping",
            "workers",
            "every_epochs",
        },
    )
    evaluation = {
        "pgd1_prediction_root": _resolved_existing_path(
            evaluation_source["pgd1_prediction_root"],
            name="evaluation.pgd1_prediction_root",
            directory=True,
        ),
        "authoritative_noisy_root": _resolved_existing_path(
            evaluation_source["authoritative_noisy_root"],
            name="evaluation.authoritative_noisy_root",
            directory=True,
        ),
        "sample_ids": _resolved_existing_path(
            evaluation_source["sample_ids"],
            name="evaluation.sample_ids",
            directory=False,
        ),
        "expected_pgd1_checkpoint_sha256": _sha256(
            evaluation_source["expected_pgd1_checkpoint_sha256"],
            name="evaluation.expected_pgd1_checkpoint_sha256",
        ),
        "expected_pgd1_config_sha256": _sha256(
            evaluation_source["expected_pgd1_config_sha256"],
            name="evaluation.expected_pgd1_config_sha256",
        ),
        "expected_pgd1_manifest_sha256": _sha256(
            evaluation_source["expected_pgd1_manifest_sha256"],
            name="evaluation.expected_pgd1_manifest_sha256",
        ),
        "expected_sample_ids_sha256": _sha256(
            evaluation_source["expected_sample_ids_sha256"],
            name="evaluation.expected_sample_ids_sha256",
        ),
        "expected_authoritative_inventory_sha256": _sha256(
            evaluation_source["expected_authoritative_inventory_sha256"],
            name="evaluation.expected_authoritative_inventory_sha256",
        ),
        "expected_sample_count": _integer(
            evaluation_source["expected_sample_count"],
            name="evaluation.expected_sample_count",
            minimum=1,
        ),
        "patch_size": _integer(
            evaluation_source["patch_size"],
            name="evaluation.patch_size",
            minimum=1,
        ),
        "seed_k": _real(
            evaluation_source["seed_k"], name="evaluation.seed_k", positive=True
        ),
        "patch_batch_size": _integer(
            evaluation_source["patch_batch_size"],
            name="evaluation.patch_batch_size",
            minimum=1,
        ),
        "niters": _integer(
            evaluation_source["niters"], name="evaluation.niters", minimum=1
        ),
        "normalization_mode": str(evaluation_source["normalization_mode"]),
        "fusion_mode": str(evaluation_source["fusion_mode"]),
        "iteration_damping": _real(
            evaluation_source["iteration_damping"],
            name="evaluation.iteration_damping",
            positive=True,
        ),
        "workers": _integer(
            evaluation_source["workers"], name="evaluation.workers", minimum=1
        ),
        "every_epochs": _integer(
            evaluation_source["every_epochs"],
            name="evaluation.every_epochs",
            minimum=1,
        ),
    }
    if evaluation["patch_size"] != model["patch_size"]:
        raise ValueError("evaluation.patch_size must equal model.patch_size")
    if evaluation["normalization_mode"] != "noisy_max":
        raise ValueError("README PGD2 evaluation requires noisy_max normalization")
    if evaluation["fusion_mode"] != "hard_best":
        raise ValueError("README PGD2 evaluation requires hard_best fusion")
    if evaluation["niters"] != 1 or evaluation["iteration_damping"] != 1.0:
        raise ValueError("README PGD2 evaluation requires niters=1 and damping=1")
    if evaluation["every_epochs"] != 1:
        raise ValueError("README PGD2 experiment requires val200 every epoch")
    if evaluation["expected_pgd1_checkpoint_sha256"] != data["expected_pgd1_checkpoint_sha256"]:
        raise ValueError("training and validation must bind the same PGD1 checkpoint")
    if evaluation["expected_pgd1_config_sha256"] != data["expected_pgd1_config_sha256"]:
        raise ValueError("training and validation must bind the same PGD1 config")

    return {
        "format": PGD2_CONFIG_FORMAT,
        "model": model,
        "data": data,
        "training": training,
        "evaluation": evaluation,
    }


def load_pgd2_config(
    path: os.PathLike[str] | str,
    *,
    require_evaluation: bool = True,
) -> tuple[dict[str, object], str]:
    config_path = Path(path).resolve()
    payload = _snapshot_regular_file(config_path)
    try:
        raw = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid PGD2 config: {config_path}") from error
    return (
        validate_pgd2_training_config(
            raw,
            require_evaluation=require_evaluation,
        ),
        hashlib.sha256(payload).hexdigest(),
    )


def pgd2_model_config_sha256(config: Mapping[str, object]) -> str:
    return canonical_config_sha256(
        {"format": "pcdenoise_pgd2_model_init_v1", "model": config["model"]}
    )


def _snapshot_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(_snapshot_regular_file(path)).hexdigest()


def _verify_resume_checkpoint_inventory(
    checkpoint_dir: Path,
    *,
    resume_checkpoint: Path,
    steps_per_epoch: int,
) -> dict[int, str]:
    """Require a contiguous, sidecar-verified inventory and the latest resume."""

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    checkpoint_paths: dict[int, Path] = {}
    sidecar_paths: dict[int, Path] = {}
    interrupted_temporaries: list[Path] = []
    for path in checkpoint_dir.iterdir():
        name = path.name
        if name.startswith(".epoch_") and name.endswith(".tmp"):
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"unexpected checkpoint artifact: {path}")
            interrupted_temporaries.append(path)
            continue
        checkpoint_name = name[:-7] if name.endswith(".sha256") else name
        match = _EPOCH_CHECKPOINT_PATTERN.fullmatch(checkpoint_name)
        if match is None or not path.is_file() or path.is_symlink():
            raise ValueError(f"unexpected checkpoint artifact: {path}")
        epoch_number = int(match.group(1))
        encoded_step = int(match.group(2))
        if epoch_number < 1 or encoded_step != epoch_number * steps_per_epoch:
            raise ValueError(f"checkpoint filename cursor is invalid: {path}")
        destination = sidecar_paths if name.endswith(".sha256") else checkpoint_paths
        if epoch_number in destination:
            raise ValueError(f"duplicate checkpoint epoch artifact: {path}")
        destination[epoch_number] = path
    orphan_epochs = set(checkpoint_paths) ^ set(sidecar_paths)
    interrupted_artifacts = interrupted_temporaries + [
        path
        for epoch_number in sorted(orphan_epochs)
        for path in (
            checkpoint_paths.get(epoch_number),
            sidecar_paths.get(epoch_number),
        )
        if path is not None
    ]
    if interrupted_artifacts:
        archive = (
            checkpoint_dir.parent
            / "recovery_orphans"
            / f"interrupted_checkpoint_publish_{time.time_ns()}"
        )
        archive.mkdir(parents=True)
        for path in interrupted_artifacts:
            os.replace(path, archive / path.name)
        for epoch_number in orphan_epochs:
            checkpoint_paths.pop(epoch_number, None)
            sidecar_paths.pop(epoch_number, None)
    epochs = sorted(checkpoint_paths)
    if not epochs or epochs != list(range(1, epochs[-1] + 1)):
        raise ValueError("checkpoint epochs are not a complete contiguous prefix")
    latest = checkpoint_paths[epochs[-1]].resolve()
    if resume_checkpoint.resolve() != latest:
        raise ValueError(f"resume checkpoint must be latest complete epoch: {latest}")
    result: dict[int, str] = {}
    for epoch_number in epochs:
        claimed = _snapshot_regular_file(sidecar_paths[epoch_number])
        try:
            claimed_sha256 = claimed.decode("ascii").strip()
        except UnicodeError as error:
            raise ValueError("checkpoint sidecar is not ASCII") from error
        _sha256(claimed_sha256, name="checkpoint sidecar SHA256")
        actual_sha256 = _file_sha256(checkpoint_paths[epoch_number])
        if claimed != (actual_sha256 + "\n").encode("ascii"):
            raise ValueError("checkpoint sidecar SHA256 mismatch")
        result[epoch_number] = actual_sha256
    return result


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _exclusive_run_lock(run_path: Path):
    """Hold one non-blocking host-local lock for the complete run operation."""

    parent = run_path.parent
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    lock_path = parent / f".{run_path.name}.pgd2_training.lock"
    stream = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            stream.seek(0)
            owner = stream.read().strip() or "unknown owner"
            raise RuntimeError(
                f"PGD2 run is already locked: {run_path}; {owner}"
            ) from error
        stream.seek(0)
        stream.truncate()
        json.dump(
            {
                "format": "pcdenoise_pgd2_run_lock_v1",
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "run_dir": str(run_path),
                "acquired_unix_time": time.time(),
            },
            stream,
            allow_nan=False,
            sort_keys=True,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        yield lock_path
    finally:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


def _sample_ids(path: Path, *, expected_count: int) -> tuple[str, ...]:
    values = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(values) != expected_count or len(set(values)) != len(values):
        raise ValueError("evaluation sample_ids count/uniqueness is invalid")
    if tuple(sorted(values)) != values:
        raise ValueError("evaluation sample_ids must be sorted")
    return values


def _preflight_validation(config: Mapping[str, object]) -> dict[str, object]:
    evaluation = config["evaluation"]
    assert isinstance(evaluation, Mapping)
    prediction_root = Path(str(evaluation["pgd1_prediction_root"]))
    manifest_path = prediction_root / "inference_manifest.json"
    payload = _snapshot_regular_file(manifest_path)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != evaluation["expected_pgd1_manifest_sha256"]:
        raise ValueError("val200 PGD1 inference manifest SHA256 mismatch")
    manifest = json.loads(payload)
    reference = manifest.get("model_reference")
    if (
        manifest.get("format") != "pcdenoise_prediction_v1"
        or manifest.get("status") != "completed"
        or manifest.get("sample_count") != evaluation["expected_sample_count"]
        or not isinstance(reference, Mapping)
        or reference.get("checkpoint_sha256")
        != evaluation["expected_pgd1_checkpoint_sha256"]
        or reference.get("config_sha256")
        != evaluation["expected_pgd1_config_sha256"]
    ):
        raise ValueError("val200 PGD1 inference binding/status is invalid")
    ids = _sample_ids(
        Path(str(evaluation["sample_ids"])),
        expected_count=int(evaluation["expected_sample_count"]),
    )
    sample_ids_sha256 = _file_sha256(Path(str(evaluation["sample_ids"])))
    if sample_ids_sha256 != evaluation["expected_sample_ids_sha256"]:
        raise ValueError("val200 sample_ids SHA256 mismatch")
    if tuple(manifest.get("sample_ids", ())) != ids:
        raise ValueError("val200 PGD1 sample IDs differ from evaluation split")
    records = manifest.get("samples")
    if not isinstance(records, list) or len(records) != len(ids):
        raise ValueError("val200 PGD1 sample records are invalid")
    authoritative_root = Path(str(evaluation["authoritative_noisy_root"]))
    point_count = int(config["data"]["expected_point_count"])
    authoritative_inventory = []
    for sample_id, record in zip(ids, records):
        if not isinstance(record, Mapping) or record.get("sample_id") != sample_id:
            raise ValueError("val200 PGD1 sample ordering is invalid")
        relative = record.get("relative_path")
        if not isinstance(relative, str):
            raise ValueError("val200 PGD1 output relative path is invalid")
        output = prediction_root / relative
        if _file_sha256(output) != record.get("output_sha256"):
            raise ValueError(f"val200 PGD1 output SHA256 mismatch: {sample_id}")
        _load_finite_float32_cloud(output, point_count=point_count)
        synset, model_id = sample_id.split("/")
        authoritative = authoritative_root / "shapenet" / synset / model_id / "noisy.npy"
        clean = authoritative.with_name("clean.npy")
        mesh = authoritative.parent / "models" / "model_normalized.obj"
        noisy_sha256 = _file_sha256(authoritative)
        clean_sha256 = _file_sha256(clean)
        mesh_sha256 = _file_sha256(mesh)
        if noisy_sha256 != record.get("input_sha256"):
            raise ValueError(f"val200 authoritative noisy SHA256 mismatch: {sample_id}")
        _load_finite_float32_cloud(authoritative, point_count=point_count)
        _load_finite_float32_cloud(clean, point_count=point_count)
        authoritative_inventory.append(
            {
                "sample_id": sample_id,
                "noisy_sha256": noisy_sha256,
                "clean_sha256": clean_sha256,
                "mesh_sha256": mesh_sha256,
            }
        )
    authoritative_inventory_sha256 = _canonical_digest(
        authoritative_inventory
    )
    if (
        authoritative_inventory_sha256
        != evaluation["expected_authoritative_inventory_sha256"]
    ):
        raise ValueError("val200 authoritative clean/noisy/mesh inventory mismatch")
    return {
        "manifest_sha256": digest,
        "sample_ids_sha256": sample_ids_sha256,
        "authoritative_inventory_sha256": authoritative_inventory_sha256,
        "sample_count": len(ids),
        "sample_ids": list(ids),
    }


def _set_optimizer_learning_rate(optimizer: object, learning_rate: float) -> None:
    optimizer.lr = float(learning_rate)
    for group in optimizer.param_groups:
        if "lr" in group:
            group["lr"] = float(learning_rate)


def _gradient_norm_and_step(optimizer: object, loss: jt.Var) -> float:
    """Measure the exact pre-update global L2 norm, then perform one Adam step."""

    optimizer.zero_grad()
    optimizer.backward(loss)
    squared_terms: list[jt.Var] = []
    for group in optimizer.param_groups:
        for parameter, gradient in zip(group["params"], group["grads"]):
            if not parameter.is_stop_grad():
                squared_terms.append((gradient.float32() ** 2).sum())
    if not squared_terms:
        raise RuntimeError("optimizer has no trainable gradients")
    squared = squared_terms[0]
    for term in squared_terms[1:]:
        squared = squared + term
    norm = float(jt.sqrt(squared).item())
    if not math.isfinite(norm):
        raise RuntimeError("gradient norm is non-finite")
    optimizer.step()
    return norm


def _displacement_statistics(displacement: jt.Var) -> tuple[float, float, float]:
    radii = jt.sqrt(
        jt.maximum(
            (displacement.float32() ** 2).sum(dim=-1),
            jt.ones(displacement.shape[:-1], dtype="float32") * 1.0e-24,
        )
    ).reshape((-1,))
    count = int(radii.shape[0])
    sorted_values = jt.sort(radii)[0]
    p95_index = max(0, min(count - 1, int(math.ceil(0.95 * count)) - 1))
    host = np.asarray(
        jt.stack((radii.mean(), sorted_values[p95_index], radii.max())).numpy(),
        dtype=np.float64,
    ).reshape(-1)
    if host.shape != (3,) or not np.isfinite(host).all():
        raise RuntimeError("displacement statistics are invalid")
    return float(host[0]), float(host[1]), float(host[2])


_STEP_METRIC_KEYS = (
    "total_loss",
    "infocd_loss_raw",
    "infocd_loss_weighted",
    "point_mse_raw",
    "point_mse_weighted",
    "correspondence_huber_loss_raw",
    "correspondence_huber_loss_weighted",
    "commitment_loss",
    "gradient_norm",
    "displacement_mean",
    "displacement_p95",
    "displacement_max",
    "learning_rate",
)


def summarize_epoch_step_metrics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not rows:
        raise ValueError("epoch metric rows must be nonempty")
    batch_sizes = np.asarray(
        [
            _integer(row.get("batch_size"), name="epoch metric batch_size", minimum=1)
            for row in rows
        ],
        dtype=np.float64,
    )
    summary: dict[str, object] = {
        "step_count": len(rows),
        "patch_count": int(batch_sizes.sum()),
    }
    for key in _STEP_METRIC_KEYS:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"epoch metric {key} is non-finite")
        summary[f"{key}_patch_weighted_mean"] = float(
            np.average(values, weights=batch_sizes)
        )
        summary[f"{key}_step_mean"] = float(values.mean())
        summary[f"{key}_step_median"] = float(np.median(values))
        summary[f"{key}_step_min"] = float(values.min())
        summary[f"{key}_step_max"] = float(values.max())
    return summary


def _ordered_batches(
    dataset: PGD2PairedDataset,
    plan: PGD2EpochPlan,
    *,
    batch_size: int,
    patch_size: int,
    prefetch_batches: int,
):
    batches = pgd2_epoch_batches(plan, batch_size=batch_size)
    if prefetch_batches == 0:
        for batch_index in range(len(batches)):
            yield batch_index, build_pgd2_training_batch(
                dataset,
                plan,
                batch_index=batch_index,
                batch_size=batch_size,
                patch_size=patch_size,
            )
        return
    with ThreadPoolExecutor(max_workers=prefetch_batches) as executor:
        futures: dict[int, Future[object]] = {}
        next_submit = 0
        while next_submit < min(prefetch_batches, len(batches)):
            futures[next_submit] = executor.submit(
                build_pgd2_training_batch,
                dataset,
                plan,
                batch_index=next_submit,
                batch_size=batch_size,
                patch_size=patch_size,
            )
            next_submit += 1
        for batch_index in range(len(batches)):
            future = futures.pop(batch_index)
            batch = future.result()
            if next_submit < len(batches):
                futures[next_submit] = executor.submit(
                    build_pgd2_training_batch,
                    dataset,
                    plan,
                    batch_index=next_submit,
                    batch_size=batch_size,
                    patch_size=patch_size,
                )
                next_submit += 1
            yield batch_index, batch


def _source_bindings(repo_root: Path) -> dict[str, str]:
    paths = {
        "train_cli": repo_root / "scripts/train_pgd2.py",
        "pgd2_runner": Path(__file__).resolve(),
        "pgd2_training_data": repo_root / "pcdenoise/data/pgd2_training.py",
        "pgd2_paired_cache": repo_root / "pcdenoise/data/pgd2_paired_cache.py",
        "surface_cache": repo_root / "pcdenoise/data/surface_cache.py",
        "noise": repo_root / "pcdenoise/data/noise.py",
        "pgd2_schedule": repo_root / "pcdenoise/training/pgd2_schedule.py",
        "prediction": repo_root / "pcdenoise/prediction.py",
        "inference": repo_root / "pcdenoise/inference.py",
        "mesh_dataset": repo_root / "pcdenoise/data/mesh_dataset.py",
        "data_archive": repo_root / "pcdenoise/data/archive.py",
        "validation_cache": repo_root / "pcdenoise/data/validation_cache.py",
        "model_factory": repo_root / "pcdenoise/models/factory.py",
        "model": repo_root / "pcdenoise/models/pgd.py",
        "model_blocks": repo_root / "pcdenoise/models/blocks.py",
        "model_conditioning": repo_root / "pcdenoise/models/conditioning.py",
        "model_vq": repo_root / "pcdenoise/models/vq.py",
        "losses": repo_root / "pcdenoise/models/losses.py",
        "ops_indexing": repo_root / "pcdenoise/ops/indexing.py",
        "ops_knn": repo_root / "pcdenoise/ops/knn.py",
        "ops_cuda_knn": repo_root / "pcdenoise/ops/cuda_knn.py",
        "ops_fps": repo_root / "pcdenoise/ops/fps.py",
        "ops_cuda_fps": repo_root / "pcdenoise/ops/cuda_fps.py",
        "ops_interpolate": repo_root / "pcdenoise/ops/interpolate.py",
        "checkpoint": repo_root / "pcdenoise/training/checkpoint.py",
        "training_engine": repo_root / "pcdenoise/training/engine.py",
        "training_logger": repo_root / "pcdenoise/training/logger.py",
        "visualization": repo_root / "pcdenoise/visualization.py",
        "evaluator": repo_root / "scripts/evaluate.py",
        "metric_chamfer": repo_root / "pcdenoise/metrics/chamfer.py",
        "metric_p2s": repo_root / "pcdenoise/metrics/p2s.py",
        "metric_competition": repo_root / "pcdenoise/metrics/competition.py",
    }
    return {name: _file_sha256(path) for name, path in paths.items()}


def _runtime_bindings() -> dict[str, object]:
    gpu_inventory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,vbios_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip().splitlines()
    return {
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "jittor": getattr(jt, "__version__", "unknown"),
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "point_cloud_utils": importlib.metadata.version("point-cloud-utils"),
        "platform": platform.platform(),
        "gpu_inventory": gpu_inventory,
        "runtime_environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_HOME",
                "CUDA_VISIBLE_DEVICES",
                "JITTOR_HOME",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "PYTHONHASHSEED",
                "cc_path",
                "conv_opt",
                "cuda_archs",
                "nvcc_path",
            )
        },
    }


def _assert_runtime_unchanged(
    repo_root: Path,
    *,
    expected_sources: Mapping[str, str],
    expected_runtime: Mapping[str, object],
) -> None:
    if _source_bindings(repo_root) != dict(expected_sources):
        raise RuntimeError("PGD2 bound source files changed during the run")
    if _runtime_bindings() != dict(expected_runtime):
        raise RuntimeError("PGD2 runtime/environment changed during the run")


def _append_jsonl(stream: TextIO, value: Mapping[str, object]) -> None:
    stream.write(json.dumps(value, allow_nan=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        payload = _snapshot_regular_file(path)
    except FileNotFoundError:
        return []
    rows: list[dict[str, object]] = []
    lines = payload.splitlines()
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index + 1 == len(lines) and not payload.endswith(b"\n"):
                break
            raise ValueError(f"invalid JSONL record in {path}")
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record must be an object: {path}")
        rows.append(value)
    return rows


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                _append_jsonl(stream, row)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _reconcile_jsonl_prefix(
    path: Path,
    *,
    cursor_key: str,
    maximum_cursor: int,
) -> list[dict[str, object]]:
    rows = _read_jsonl(path)
    retained: list[dict[str, object]] = []
    previous = 0
    for row in rows:
        cursor = _integer(row.get(cursor_key), name=f"{path.name} {cursor_key}")
        if cursor <= previous:
            raise ValueError(f"{path.name} cursors are not strictly increasing")
        previous = cursor
        if cursor <= maximum_cursor:
            retained.append(row)
    retained_cursors = [int(row[cursor_key]) for row in retained]
    if retained_cursors != list(range(1, len(retained) + 1)):
        raise ValueError(f"{path.name} does not contain a contiguous prefix")
    if retained != rows or (path.exists() and not _snapshot_regular_file(path).endswith(b"\n")):
        _atomic_jsonl(path, retained)
    return retained


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _load_finite_float32_cloud(path: Path, *, point_count: int) -> np.ndarray:
    payload = _snapshot_regular_file(path)
    values = np.load(io.BytesIO(payload), allow_pickle=False)
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.float32
        or values.shape != (point_count, 3)
        or not np.isfinite(values).all()
    ):
        raise ValueError(f"invalid float32 point-cloud artifact: {path}")
    return np.ascontiguousarray(values)


def _verify_existing_prediction(
    prediction_dir: Path,
    *,
    config: Mapping[str, object],
    config_sha256: str,
    checkpoint: Mapping[str, object],
    sample_ids: Sequence[str],
) -> dict[str, object]:
    """Strictly reconcile one already-published PGD2 prediction directory."""

    evaluation = config["evaluation"]
    assert isinstance(evaluation, Mapping)
    manifest_path = prediction_dir / "inference_manifest.json"
    manifest = json.loads(_snapshot_regular_file(manifest_path))
    reference = manifest.get("model_reference")
    if (
        manifest.get("format") != "pcdenoise_prediction_v1"
        or manifest.get("format_version") != 1
        or manifest.get("status") != "completed"
        or manifest.get("sample_count") != len(sample_ids)
        or manifest.get("sample_ids") != list(sample_ids)
        or manifest.get("input_filename") != "denoised.npy"
        or manifest.get("authoritative_input_filename") != "noisy.npy"
        or manifest.get("patch_size") != evaluation["patch_size"]
        or float(manifest.get("seed_k")) != float(evaluation["seed_k"])
        or manifest.get("patch_batch_size") != evaluation["patch_batch_size"]
        or manifest.get("niters") != evaluation["niters"]
        or manifest.get("normalization_mode")
        != evaluation["normalization_mode"]
        or manifest.get("fusion_mode") != evaluation["fusion_mode"]
        or float(manifest.get("iteration_damping"))
        != float(evaluation["iteration_damping"])
        or not isinstance(reference, Mapping)
        or reference.get("checkpoint_sha256")
        != checkpoint["checkpoint_sha256"]
        or reference.get("checkpoint_step") != checkpoint["step"]
        or reference.get("config_sha256") != config_sha256
        or reference.get("architecture") != "pgd"
        or reference.get("pgd2_stage") != 2
    ):
        raise ValueError("existing PGD2 prediction manifest binding is invalid")
    records = manifest.get("samples")
    if not isinstance(records, list) or len(records) != len(sample_ids):
        raise ValueError("existing PGD2 prediction sample ledger is invalid")
    pgd1_root = Path(str(evaluation["pgd1_prediction_root"]))
    authoritative_root = Path(str(evaluation["authoritative_noisy_root"]))
    point_count = int(config["data"]["expected_point_count"])
    authoritative_inventory = []
    prediction_inventory = []
    for sample_id, record in zip(sample_ids, records):
        if not isinstance(record, Mapping) or record.get("sample_id") != sample_id:
            raise ValueError("existing PGD2 prediction ordering is invalid")
        synset, model_id = sample_id.split("/")
        expected_relative = (
            Path("shapenet") / synset / model_id / "denoised.npy"
        ).as_posix()
        if (
            record.get("relative_path") != expected_relative
            or record.get("point_count") != point_count
        ):
            raise ValueError("existing PGD2 prediction layout/count is invalid")
        model_input = pgd1_root / "shapenet" / synset / model_id / "denoised.npy"
        authoritative = (
            authoritative_root / "shapenet" / synset / model_id / "noisy.npy"
        )
        output = prediction_dir / expected_relative
        input_sha256 = _file_sha256(model_input)
        authoritative_sha256 = _file_sha256(authoritative)
        output_sha256 = _file_sha256(output)
        if (
            record.get("input_sha256") != input_sha256
            or record.get("authoritative_input_sha256")
            != authoritative_sha256
            or record.get("output_sha256") != output_sha256
        ):
            raise ValueError("existing PGD2 prediction SHA256 binding is invalid")
        _load_finite_float32_cloud(output, point_count=point_count)
        authoritative_inventory.append(
            {
                "sample_id": sample_id,
                "authoritative_input_sha256": authoritative_sha256,
            }
        )
        prediction_inventory.append(
            {"shape_id": sample_id, "sha256": output_sha256}
        )
    if manifest.get("authoritative_input_inventory_sha256") != _canonical_digest(
        authoritative_inventory
    ):
        raise ValueError("existing authoritative input inventory is invalid")
    return {
        "sample_count": len(sample_ids),
        "shape_ids": list(sample_ids),
        "content_sha256": _canonical_digest(prediction_inventory),
    }


def _load_existing_evaluation(
    evaluation_dir: Path,
    *,
    sample_ids: Sequence[str],
    prediction_inventory: Mapping[str, object],
    expected_evaluator_source_sha256: str,
) -> dict[str, object]:
    manifest_path = evaluation_dir / "evaluation_manifest.json"
    manifest = json.loads(_snapshot_regular_file(manifest_path))
    claimed_content = manifest.get("content_sha256")
    unsigned = dict(manifest)
    unsigned.pop("content_sha256", None)
    metrics_path = evaluation_dir / "metrics_summary.json"
    csv_path = evaluation_dir / "metrics_per_sample.csv"
    if (
        manifest.get("format") != "pcdenoise_evaluation_v1"
        or manifest.get("format_version") != 1
        or manifest.get("completed") is not True
        or manifest.get("sample_count") != len(sample_ids)
        or manifest.get("valid_sample_count") != len(sample_ids)
        or manifest.get("sample_ids") != list(sample_ids)
        or manifest.get("require_p2s") is not True
        or manifest.get("prediction_inventory") != dict(prediction_inventory)
        or manifest.get("evaluator_source_sha256")
        != expected_evaluator_source_sha256
        or claimed_content != _canonical_digest(unsigned)
        or manifest.get("artifacts")
        != {
            "metrics_per_sample_csv_sha256": _file_sha256(csv_path),
            "metrics_summary_json_sha256": _file_sha256(metrics_path),
        }
    ):
        raise ValueError("existing PGD2 evaluation binding is invalid")
    metrics = json.loads(_snapshot_regular_file(metrics_path))
    if (
        metrics.get("sample_count") != len(sample_ids)
        or metrics.get("valid_count") != len(sample_ids)
        or metrics.get("missing_count") != 0
    ):
        raise ValueError("existing PGD2 evaluation sample counts are invalid")
    for key in (
        "mean_cd_score",
        "mean_p2s_score",
        "total_score",
        "mean_cd_pred",
        "mean_p2s_pred",
    ):
        if not math.isfinite(float(metrics[key])):
            raise ValueError(f"existing PGD2 evaluation {key} is non-finite")
    return metrics


def _run_validation(
    *,
    model: object,
    config: Mapping[str, object],
    config_sha256: str,
    checkpoint: Mapping[str, object],
    epoch_number: int,
    run_dir: Path,
    repo_root: Path,
) -> dict[str, object]:
    evaluation = config["evaluation"]
    assert isinstance(evaluation, Mapping)
    # Re-pin the complete PGD1/noisy/clean/mesh validation inventory before
    # every score, not merely once when a multi-day run starts.
    _preflight_validation(config)
    evaluator_source_sha256 = _file_sha256(repo_root / "scripts/evaluate.py")
    prediction_dir = run_dir / "predictions" / f"epoch_{epoch_number:04d}"
    evaluation_dir = run_dir / "evaluations" / f"epoch_{epoch_number:04d}"
    ids = _sample_ids(
        Path(str(evaluation["sample_ids"])),
        expected_count=int(evaluation["expected_sample_count"]),
    )
    model.eval()
    if prediction_dir.exists():
        prediction_inventory = _verify_existing_prediction(
            prediction_dir,
            config=config,
            config_sha256=config_sha256,
            checkpoint=checkpoint,
            sample_ids=ids,
        )
    else:
        run_prediction(
            model,
            input_root=Path(str(evaluation["pgd1_prediction_root"])),
            input_filename="denoised.npy",
            authoritative_input_root=Path(
                str(evaluation["authoritative_noisy_root"])
            ),
            authoritative_input_filename="noisy.npy",
            output_dir=prediction_dir,
            patch_size=int(evaluation["patch_size"]),
            seed_k=float(evaluation["seed_k"]),
            patch_batch_size=int(evaluation["patch_batch_size"]),
            niters=int(evaluation["niters"]),
            normalization_mode=str(evaluation["normalization_mode"]),
            fusion_mode=str(evaluation["fusion_mode"]),
            iteration_damping=float(evaluation["iteration_damping"]),
            sample_ids=ids,
            model_reference={
                "checkpoint": str(checkpoint["path"]),
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "checkpoint_step": checkpoint["step"],
                "config_sha256": config_sha256,
                "architecture": pgd_model_architecture(config["model"]),
                "pgd2_stage": 2,
            },
        )
        prediction_inventory = _verify_existing_prediction(
            prediction_dir,
            config=config,
            config_sha256=config_sha256,
            checkpoint=checkpoint,
            sample_ids=ids,
        )
    command = [
        sys.executable,
        "-u",
        str(repo_root / "scripts/evaluate.py"),
        "--pred-dir",
        str(prediction_dir),
        "--gt-dir",
        str(evaluation["authoritative_noisy_root"]),
        "--noisy-dir",
        str(evaluation["authoritative_noisy_root"]),
        "--mesh-dir",
        str(evaluation["authoritative_noisy_root"]),
        "--pred-filename",
        "denoised.npy",
        "--gt-filename",
        "clean.npy",
        "--noisy-filename",
        "noisy.npy",
        "--workers",
        str(evaluation["workers"]),
        "--sample-ids",
        str(evaluation["sample_ids"]),
        "--output-dir",
        str(evaluation_dir),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    if evaluation_dir.exists():
        metrics = _load_existing_evaluation(
            evaluation_dir,
            sample_ids=ids,
            prediction_inventory=prediction_inventory,
            expected_evaluator_source_sha256=evaluator_source_sha256,
        )
    else:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (evaluation_dir.parent / f"epoch_{epoch_number:04d}.log").write_text(
            completed.stdout, encoding="utf-8"
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"epoch {epoch_number} evaluation failed with code "
                f"{completed.returncode}: {completed.stdout[-2000:]}"
            )
        metrics = _load_existing_evaluation(
            evaluation_dir,
            sample_ids=ids,
            prediction_inventory=prediction_inventory,
            expected_evaluator_source_sha256=evaluator_source_sha256,
        )
    required = ("mean_cd_score", "mean_p2s_score", "total_score")
    for key in required:
        if not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"validation metric {key} is non-finite")
    model.train()
    return {
        "epoch": epoch_number,
        "step": int(checkpoint["step"]),
        "checkpoint_path": str(checkpoint["path"]),
        "checkpoint_sha256": str(checkpoint["checkpoint_sha256"]),
        "prediction_dir": str(prediction_dir),
        "evaluation_dir": str(evaluation_dir),
        "mean_cd_score": float(metrics["mean_cd_score"]),
        "mean_p2s_score": float(metrics["mean_p2s_score"]),
        "total_score": float(metrics["total_score"]),
        "mean_cd_pred": float(metrics["mean_cd_pred"]),
        "mean_p2s_pred": float(metrics["mean_p2s_pred"]),
    }


def _update_best(run_dir: Path, validation: Mapping[str, object]) -> bool:
    path = run_dir / "best_checkpoint.json"
    current = None
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    improved = current is None or float(validation["total_score"]) > float(
        current["total_score"]
    )
    if improved:
        _atomic_json(path, dict(validation))
    return improved


def _rebuild_best(
    run_dir: Path, validations: Sequence[Mapping[str, object]]
) -> dict[str, object] | None:
    if not validations:
        return None
    winner = max(validations, key=lambda item: float(item["total_score"]))
    canonical = {key: value for key, value in winner.items() if key != "is_best"}
    _atomic_json(run_dir / "best_checkpoint.json", canonical)
    return canonical


_VALIDATION_METRIC_KEYS = (
    "mean_cd_score",
    "mean_p2s_score",
    "total_score",
    "mean_cd_pred",
    "mean_p2s_pred",
)


def _canonicalize_validation_rows(
    run_dir: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    steps_per_epoch: int,
) -> list[dict[str, object]]:
    """Validate the durable score ledger and recompute its derived best flags."""

    result: list[dict[str, object]] = []
    best_score = -math.inf
    for epoch_number, source in enumerate(rows, start=1):
        row = dict(source)
        expected_step = epoch_number * steps_per_epoch
        if (
            _integer(row.get("epoch"), name="validation epoch", minimum=1)
            != epoch_number
            or _integer(row.get("step"), name="validation step", minimum=1)
            != expected_step
        ):
            raise ValueError("validation score cursor is not a complete epoch prefix")
        expected_checkpoint = (
            run_dir
            / "checkpoints"
            / f"epoch_{epoch_number:04d}_step_{expected_step:08d}.pkl"
        ).resolve()
        checkpoint_path = Path(str(row.get("checkpoint_path", ""))).resolve()
        if checkpoint_path != expected_checkpoint or not checkpoint_path.is_file():
            raise ValueError("validation score checkpoint path is invalid")
        if row.get("checkpoint_sha256") != _file_sha256(checkpoint_path):
            raise ValueError("validation score checkpoint SHA256 is invalid")
        for kind in ("prediction", "evaluation"):
            expected = (
                run_dir / f"{kind}s" / f"epoch_{epoch_number:04d}"
            ).resolve()
            actual = Path(str(row.get(f"{kind}_dir", ""))).resolve()
            if actual != expected or not actual.is_dir():
                raise ValueError(f"validation score {kind} directory is invalid")
        for key in _VALIDATION_METRIC_KEYS:
            if not math.isfinite(float(row.get(key))):
                raise ValueError(f"validation score {key} is non-finite")
        total_score = float(row["total_score"])
        row["is_best"] = total_score > best_score
        best_score = max(best_score, total_score)
        result.append(row)
    return result


def _verify_historical_validation_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    config: Mapping[str, object],
    config_sha256: str,
    repo_root: Path,
) -> None:
    """Re-verify every durable prediction/evaluation receipt on resume."""

    evaluation = config["evaluation"]
    assert isinstance(evaluation, Mapping)
    sample_ids = _sample_ids(
        Path(str(evaluation["sample_ids"])),
        expected_count=int(evaluation["expected_sample_count"]),
    )
    evaluator_source_sha256 = _file_sha256(repo_root / "scripts/evaluate.py")
    for row in rows:
        checkpoint = {
            "path": str(row["checkpoint_path"]),
            "checkpoint_sha256": str(row["checkpoint_sha256"]),
            "step": int(row["step"]),
        }
        prediction_inventory = _verify_existing_prediction(
            Path(str(row["prediction_dir"])),
            config=config,
            config_sha256=config_sha256,
            checkpoint=checkpoint,
            sample_ids=sample_ids,
        )
        metrics = _load_existing_evaluation(
            Path(str(row["evaluation_dir"])),
            sample_ids=sample_ids,
            prediction_inventory=prediction_inventory,
            expected_evaluator_source_sha256=evaluator_source_sha256,
        )
        for key in _VALIDATION_METRIC_KEYS:
            if float(row[key]) != float(metrics[key]):
                raise ValueError(
                    f"historical validation score differs from artifacts: {key}"
                )


def _reconcile_resume_logs(
    run_dir: Path,
    *,
    checkpoint: Mapping[str, object],
    start_epoch: int,
    global_step: int,
    manifest_completed_epochs: int,
    steps_per_epoch: int,
    patches_per_epoch: int,
    run_validation: bool = True,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Trim failed future work and prove all checkpointed log prefixes complete."""

    step_path = run_dir / "training_steps.jsonl"
    epoch_path = run_dir / "epoch_summaries.jsonl"
    score_path = run_dir / "validation_scores.jsonl"
    step_rows = _reconcile_jsonl_prefix(
        step_path, cursor_key="step", maximum_cursor=global_step
    )
    if len(step_rows) != global_step:
        raise ValueError("training step ledger is incomplete at resume checkpoint")
    for expected_step, row in enumerate(step_rows, start=1):
        expected_epoch = (expected_step - 1) // steps_per_epoch + 1
        expected_batch = (expected_step - 1) % steps_per_epoch
        if (
            _integer(row.get("epoch"), name="training step epoch", minimum=1)
            != expected_epoch
            or _integer(
                row.get("batch_index"), name="training batch index", minimum=0
            )
            != expected_batch
        ):
            raise ValueError("training step ledger epoch/batch cursor is invalid")
    epoch_rows = _reconcile_jsonl_prefix(
        epoch_path, cursor_key="epoch", maximum_cursor=start_epoch
    )
    if len(epoch_rows) != start_epoch:
        raise ValueError("epoch summary ledger is incomplete at resume checkpoint")
    for epoch_number, row in enumerate(epoch_rows, start=1):
        if (
            _integer(row.get("global_step"), name="epoch global_step", minimum=1)
            != epoch_number * steps_per_epoch
            or _integer(row.get("step_count"), name="epoch step_count", minimum=1)
            != steps_per_epoch
            or _integer(row.get("patch_count"), name="epoch patch_count", minimum=1)
            != patches_per_epoch
        ):
            raise ValueError("epoch summary ledger cursor/count is invalid")
    if start_epoch and dict(checkpoint.get("metrics") or {}) != epoch_rows[-1]:
        raise ValueError("checkpoint metrics differ from durable epoch summary")

    score_rows = _reconcile_jsonl_prefix(
        score_path, cursor_key="epoch", maximum_cursor=start_epoch
    )
    if run_validation:
        allowed_score_counts = {manifest_completed_epochs}
        if manifest_completed_epochs == start_epoch - 1:
            allowed_score_counts.add(start_epoch)
        if len(score_rows) not in allowed_score_counts:
            raise ValueError("validation score ledger disagrees with run transaction")
        score_rows = _canonicalize_validation_rows(
            run_dir, score_rows, steps_per_epoch=steps_per_epoch
        )
    elif score_rows:
        raise ValueError("validation ledger must be empty when validation is skipped")
    _atomic_jsonl(score_path, score_rows)
    if score_rows:
        _rebuild_best(run_dir, score_rows)
    elif (run_dir / "best_checkpoint.json").exists():
        archive_root = run_dir / "recovery_orphans"
        archive_root.mkdir(exist_ok=True)
        os.replace(
            run_dir / "best_checkpoint.json",
            archive_root / f"uncommitted_best_{time.time_ns()}.json",
        )
    return step_rows, epoch_rows, score_rows


def _log_training_row(
    writer: SummaryWriter,
    row: Mapping[str, object],
    *,
    steps_per_epoch: int,
    log_every: int,
) -> None:
    step = int(row["step"])
    batch_index = int(row["batch_index"])
    if step != 1 and step % log_every != 0 and batch_index + 1 != steps_per_epoch:
        return
    for key in _STEP_METRIC_KEYS:
        writer.add_scalar(f"training/{key}", row[key], step)
    writer.add_scalar("training/ema_updates", row["ema_updates"], step)


def _log_epoch_summary(writer: SummaryWriter, row: Mapping[str, object]) -> None:
    step = int(row["global_step"])
    for key, value in row.items():
        if key not in {"epoch", "global_step"}:
            writer.add_scalar(f"epoch/{key}", value, step)


def _log_validation_row(writer: SummaryWriter, row: Mapping[str, object]) -> None:
    step = int(row["step"])
    for key in _VALIDATION_METRIC_KEYS:
        writer.add_scalar(f"validation/{key}", row[key], step)
    writer.add_scalar("validation/epoch", row["epoch"], step)
    writer.add_scalar("validation/is_best", int(bool(row["is_best"])), step)


def _open_rebuilt_tensorboard(
    run_dir: Path,
    *,
    is_resume: bool,
    global_step: int,
    step_rows: Sequence[Mapping[str, object]],
    epoch_rows: Sequence[Mapping[str, object]],
    score_rows: Sequence[Mapping[str, object]],
    steps_per_epoch: int,
    log_every: int,
) -> SummaryWriter:
    """Create one unambiguous TensorBoard history, replaying durable JSONL."""

    tensorboard_dir = run_dir / "tensorboard"
    if is_resume and tensorboard_dir.exists():
        archive_root = run_dir / "tensorboard_recovery_archives"
        archive_root.mkdir(exist_ok=True)
        destination = archive_root / (
            f"before_step_{global_step:08d}_{time.time_ns()}"
        )
        os.replace(tensorboard_dir, destination)
    tensorboard_dir.mkdir(exist_ok=False)
    writer = SummaryWriter(logdir=str(tensorboard_dir), flush_secs=30)
    for row in step_rows:
        _log_training_row(
            writer,
            row,
            steps_per_epoch=steps_per_epoch,
            log_every=log_every,
        )
    for row in epoch_rows:
        _log_epoch_summary(writer, row)
    for row in score_rows:
        _log_validation_row(writer, row)
    writer.flush()
    return writer


def _run_manifest(
    *,
    status: str,
    completed_epochs: int,
    global_step: int,
    max_epochs: int,
    steps_per_epoch: int,
    config_sha256: str,
    initial_checkpoint_sha256: str,
    last_validation: Mapping[str, object] | None = None,
    error: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "format": "pcdenoise_pgd2_run_v1",
        "status": status,
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "max_epochs": max_epochs,
        "steps_per_epoch": steps_per_epoch,
        "max_steps": max_epochs * steps_per_epoch,
        "config_sha256": config_sha256,
        "initial_checkpoint_sha256": initial_checkpoint_sha256,
        "updated_unix_time": time.time(),
    }
    if last_validation is not None:
        result["last_validation"] = dict(last_validation)
    if error is not None:
        result["error"] = error
    return result


def _run_pgd2_training_unlocked(
    config: Mapping[str, object],
    *,
    raw_config_sha256: str,
    run_dir: os.PathLike[str] | str,
    initial_checkpoint: os.PathLike[str] | str | None = None,
    expected_initial_checkpoint_sha256: str | None = None,
    resume_checkpoint: os.PathLike[str] | str | None = None,
    run_validation: bool = True,
    command: Sequence[str] | None = None,
) -> dict[str, object]:
    """Train one fixed-cache PGD2 arm through its configured epoch limit."""

    canonical = validate_pgd2_training_config(
        config,
        require_evaluation=run_validation,
    )
    config_sha256 = canonical_config_sha256(canonical)
    _sha256(raw_config_sha256, name="raw_config_sha256")
    if (initial_checkpoint is None) == (resume_checkpoint is None):
        raise ValueError("supply exactly one of initial_checkpoint or resume_checkpoint")
    if initial_checkpoint is not None and expected_initial_checkpoint_sha256 is None:
        raise ValueError("new training requires expected initial checkpoint SHA256")
    if resume_checkpoint is not None and expected_initial_checkpoint_sha256 is not None:
        raise ValueError("resume must not supply an initial checkpoint SHA256")

    repo_root = Path(__file__).resolve().parents[2]
    run_path = Path(run_dir).resolve()
    is_resume = resume_checkpoint is not None
    existing_run_manifest: dict[str, object] | None = None
    if is_resume:
        if not run_path.is_dir():
            raise FileNotFoundError(run_path)
        existing_run_manifest = json.loads(
            _snapshot_regular_file(run_path / "run_manifest.json")
        )
        if not isinstance(existing_run_manifest, dict):
            raise ValueError("existing run manifest must be an object")
        resume_path = Path(resume_checkpoint).resolve()
        if resume_path.parent != (run_path / "checkpoints").resolve():
            raise ValueError("resume checkpoint must belong to this run directory")

    training = canonical["training"]
    data = canonical["data"]
    assert isinstance(training, Mapping) and isinstance(data, Mapping)
    use_cuda = bool(training["use_cuda"])
    if use_cuda and not jt.has_cuda:
        raise RuntimeError("PGD2 config requires CUDA but Jittor has no CUDA")
    jt.flags.use_cuda = int(use_cuda)
    jt.set_global_seed(int(training["seed"]))

    dataset = load_pgd2_paired_dataset(
        data["paired_cache"],
        expected_content_sha256=str(data["expected_content_sha256"]),
        expected_pgd1_checkpoint_sha256=str(
            data["expected_pgd1_checkpoint_sha256"]
        ),
        expected_pgd1_config_sha256=str(data["expected_pgd1_config_sha256"]),
        expected_point_count=int(data["expected_point_count"]),
    )
    if len(dataset.sample_ids) != int(data["expected_sample_count"]):
        raise ValueError("paired dataset sample count differs from config")
    if run_validation:
        validation_binding: dict[str, object] = _preflight_validation(canonical)
        validation_ids = tuple(
            str(item) for item in validation_binding["sample_ids"]
        )
        overlap = sorted(set(dataset.sample_ids) & set(validation_ids))
        if overlap:
            raise ValueError(
                "PGD2 training and val200 IDs must be disjoint; "
                f"overlap={overlap[:3]}"
            )
    else:
        validation_binding = {"mode": "skipped"}
    total_patches = len(dataset.sample_ids) * int(training["patches_per_shape"])
    steps_per_epoch = math.ceil(total_patches / int(training["batch_size"]))
    max_epochs = int(training["max_epochs"])
    if is_resume:
        _verify_resume_checkpoint_inventory(
            run_path / "checkpoints",
            resume_checkpoint=Path(resume_checkpoint).resolve(),
            steps_per_epoch=steps_per_epoch,
        )

    model = build_pgd_model(canonical["model"], canonical["data"])
    optimizer = jt.optim.Adam(
        model.parameters(), lr=float(training["learning_rate"])
    )
    model_config_sha256 = pgd2_model_config_sha256(canonical)
    if is_resume:
        checkpoint = load_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            expected_config_sha256=config_sha256,
        )
        state = checkpoint.get("training_state")
        if not isinstance(state, Mapping):
            raise ValueError("resume checkpoint has no PGD2 training_state")
        start_epoch = _integer(
            state.get("completed_epochs"),
            name="resume completed_epochs",
            minimum=0,
        )
        global_step = _integer(
            state.get("global_step"), name="resume global_step", minimum=0
        )
        if (
            checkpoint.get("step") != global_step
            or start_epoch > max_epochs
            or state.get("steps_per_epoch") != steps_per_epoch
            or global_step != start_epoch * steps_per_epoch
            or state.get("paired_content_sha256") != dataset.content_sha256
            or state.get("sample_ids_sha256")
            != hashlib.sha256(
                "".join(f"{item}\n" for item in dataset.sample_ids).encode("ascii")
            ).hexdigest()
        ):
            raise ValueError("resume checkpoint PGD2 cursor/data binding is invalid")
        optimizer_step = getattr(optimizer, "n_step", None)
        if optimizer_step is None or int(optimizer_step) != global_step:
            raise ValueError("resume optimizer Adam step differs from global_step")
        initial_digest = _sha256(
            state.get("initial_checkpoint_sha256"),
            name="resume initial checkpoint SHA256",
        )
        assert existing_run_manifest is not None
        manifest_completed_epochs = _integer(
            existing_run_manifest.get("completed_epochs"),
            name="run manifest completed_epochs",
            minimum=0,
        )
        if (
            existing_run_manifest.get("format") != "pcdenoise_pgd2_run_v1"
            or existing_run_manifest.get("config_sha256") != config_sha256
            or existing_run_manifest.get("max_epochs") != max_epochs
            or existing_run_manifest.get("steps_per_epoch") != steps_per_epoch
            or existing_run_manifest.get("initial_checkpoint_sha256")
            != initial_digest
            or manifest_completed_epochs not in {start_epoch - 1, start_epoch}
        ):
            raise ValueError("existing run manifest and resume checkpoint disagree")
        pending_validation_epoch = (
            start_epoch
            if run_validation and manifest_completed_epochs == start_epoch - 1
            else None
        )
    else:
        expected_init = _sha256(
            expected_initial_checkpoint_sha256,
            name="expected initial checkpoint SHA256",
        )
        checkpoint = load_checkpoint(
            initial_checkpoint,
            model=model,
            expected_config_sha256=model_config_sha256,
        )
        if checkpoint["checkpoint_sha256"] != expected_init or checkpoint["step"] != 0:
            raise ValueError("initial checkpoint SHA256/step binding mismatch")
        initial_digest = expected_init
        start_epoch = 0
        global_step = 0
        manifest_completed_epochs = 0
        pending_validation_epoch = None

    # A new destination becomes visible only after every immutable input and
    # the common initialization checkpoint have passed preflight.  This keeps
    # a typo or tampered cache from leaving an unusable half-created run.
    if is_resume:
        for name in ("checkpoints", "evaluations", "predictions"):
            path = run_path / name
            if not path.is_dir():
                raise FileNotFoundError(path)
    else:
        run_path.mkdir(parents=True, exist_ok=False)
        for name in ("checkpoints", "evaluations", "predictions"):
            (run_path / name).mkdir()

    parameters_path = run_path / "training_parameters.json"
    source_bindings = _source_bindings(repo_root)
    runtime_binding = _runtime_bindings()
    if not is_resume:
        _atomic_json(
            parameters_path,
            {
                "format": "pcdenoise_pgd2_training_parameters_v1",
                "config": canonical,
                "raw_config_sha256": raw_config_sha256,
                "canonical_config_sha256": config_sha256,
                "model_config_sha256": model_config_sha256,
                "initial_checkpoint": str(Path(initial_checkpoint).resolve()),
                "initial_checkpoint_sha256": initial_digest,
                "paired_cache_content_sha256": dataset.content_sha256,
                "paired_sample_count": len(dataset.sample_ids),
                "paired_point_count": dataset.point_count,
                "steps_per_epoch": steps_per_epoch,
                "patches_per_epoch": total_patches,
                "validation": validation_binding,
                "source_sha256": source_bindings,
                "command": list(command or ()),
                **runtime_binding,
            },
        )
    else:
        parameters = json.loads(_snapshot_regular_file(parameters_path))
        if (
            parameters.get("format")
            != "pcdenoise_pgd2_training_parameters_v1"
            or parameters.get("raw_config_sha256") != raw_config_sha256
            or parameters.get("canonical_config_sha256") != config_sha256
            or parameters.get("model_config_sha256") != model_config_sha256
            or parameters.get("initial_checkpoint_sha256") != initial_digest
            or parameters.get("paired_cache_content_sha256")
            != dataset.content_sha256
            or parameters.get("paired_sample_count") != len(dataset.sample_ids)
            or parameters.get("paired_point_count") != dataset.point_count
            or parameters.get("steps_per_epoch") != steps_per_epoch
            or parameters.get("patches_per_epoch") != total_patches
            or parameters.get("validation") != validation_binding
            or parameters.get("source_sha256") != source_bindings
            or parameters.get("python") != runtime_binding["python"]
            or parameters.get("python_executable")
            != runtime_binding["python_executable"]
            or parameters.get("jittor") != runtime_binding["jittor"]
            or parameters.get("numpy") != runtime_binding["numpy"]
            or parameters.get("scipy") != runtime_binding["scipy"]
            or parameters.get("point_cloud_utils")
            != runtime_binding["point_cloud_utils"]
            or parameters.get("platform") != runtime_binding["platform"]
            or parameters.get("gpu_inventory")
            != runtime_binding["gpu_inventory"]
            or parameters.get("runtime_environment")
            != runtime_binding["runtime_environment"]
        ):
            raise ValueError(
                "existing run training/source/environment bindings differ on resume"
            )

    step_log_path = run_path / "training_steps.jsonl"
    epoch_log_path = run_path / "epoch_summaries.jsonl"
    scores_path = run_path / "validation_scores.jsonl"
    if is_resume:
        step_rows, epoch_rows, score_rows = _reconcile_resume_logs(
            run_path,
            checkpoint=checkpoint,
            start_epoch=start_epoch,
            global_step=global_step,
            manifest_completed_epochs=manifest_completed_epochs,
            steps_per_epoch=steps_per_epoch,
            patches_per_epoch=total_patches,
            run_validation=run_validation,
        )
        if run_validation:
            _verify_historical_validation_rows(
                score_rows,
                config=canonical,
                config_sha256=config_sha256,
                repo_root=repo_root,
            )
    else:
        step_rows, epoch_rows, score_rows = [], [], []
    completed_epochs = manifest_completed_epochs
    last_validation = score_rows[-1] if score_rows else None
    writer = _open_rebuilt_tensorboard(
        run_path,
        is_resume=is_resume,
        global_step=global_step,
        step_rows=step_rows,
        epoch_rows=epoch_rows,
        score_rows=score_rows,
        steps_per_epoch=steps_per_epoch,
        log_every=int(training["log_every"]),
    )
    _atomic_json(
        run_path / "run_manifest.json",
        _run_manifest(
            status="running",
            completed_epochs=completed_epochs,
            global_step=global_step,
            max_epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
            config_sha256=config_sha256,
            initial_checkpoint_sha256=initial_digest,
            last_validation=last_validation,
        ),
    )

    model.train()
    try:
        with (
            step_log_path.open("a", encoding="utf-8", buffering=1) as step_stream,
            epoch_log_path.open("a", encoding="utf-8", buffering=1) as epoch_stream,
            scores_path.open("a", encoding="utf-8", buffering=1) as score_stream,
        ):
            if pending_validation_epoch is not None:
                _assert_runtime_unchanged(
                    repo_root,
                    expected_sources=source_bindings,
                    expected_runtime=runtime_binding,
                )
                recovered_validation = _run_validation(
                    model=model,
                    config=canonical,
                    config_sha256=config_sha256,
                    checkpoint=checkpoint,
                    epoch_number=pending_validation_epoch,
                    run_dir=run_path,
                    repo_root=repo_root,
                )
                _assert_runtime_unchanged(
                    repo_root,
                    expected_sources=source_bindings,
                    expected_runtime=runtime_binding,
                )
                if len(score_rows) == pending_validation_epoch:
                    existing_validation = dict(score_rows[-1])
                    existing_payload = {
                        key: value
                        for key, value in existing_validation.items()
                        if key != "is_best"
                    }
                    if existing_payload != recovered_validation:
                        raise ValueError(
                            "recovered validation differs from durable score row"
                        )
                    last_validation = existing_validation
                else:
                    recovered_validation["is_best"] = _update_best(
                        run_path, recovered_validation
                    )
                    _append_jsonl(score_stream, recovered_validation)
                    score_stream.flush()
                    os.fsync(score_stream.fileno())
                    score_rows.append(recovered_validation)
                    last_validation = recovered_validation
                    _log_validation_row(writer, recovered_validation)
                completed_epochs = pending_validation_epoch
                _atomic_json(
                    run_path / "run_manifest.json",
                    _run_manifest(
                        status="running",
                        completed_epochs=completed_epochs,
                        global_step=global_step,
                        max_epochs=max_epochs,
                        steps_per_epoch=steps_per_epoch,
                        config_sha256=config_sha256,
                        initial_checkpoint_sha256=initial_digest,
                        last_validation=last_validation,
                    ),
                )
                writer.flush()

            for epoch_index in range(start_epoch, max_epochs):
                _assert_runtime_unchanged(
                    repo_root,
                    expected_sources=source_bindings,
                    expected_runtime=runtime_binding,
                )
                epoch_started = time.perf_counter()
                plan = build_pgd2_epoch_plan(
                    dataset,
                    seed=int(training["seed"]),
                    epoch=epoch_index,
                    patches_per_shape=int(training["patches_per_shape"]),
                )
                epoch_rows: list[dict[str, object]] = []
                for batch_index, batch in _ordered_batches(
                    dataset,
                    plan,
                    batch_size=int(training["batch_size"]),
                    patch_size=int(canonical["model"]["patch_size"]),
                    prefetch_batches=int(training["prefetch_batches"]),
                ):
                    global_step = epoch_index * steps_per_epoch + batch_index + 1
                    learning_rate = pgd2_learning_rate_at_step(
                        {
                            "learning_rate": training["learning_rate"],
                            "max_epochs": training["max_epochs"],
                            "learning_rate_schedule": training[
                                "learning_rate_schedule"
                            ],
                        },
                        absolute_step=global_step,
                        steps_per_epoch=steps_per_epoch,
                    )
                    _set_optimizer_learning_rate(optimizer, learning_rate)
                    inputs = jt.array(batch.pgd1_input)
                    targets = jt.array(batch.clean_target)
                    prediction, details = model(inputs, return_details=True)
                    infocd = infocd_loss(
                        prediction,
                        targets,
                        profile=str(training["loss_profile"]),
                    )
                    point_mse = ((prediction - targets) ** 2).mean()
                    huber = correspondence_huber_loss(
                        prediction,
                        targets,
                        delta=float(training["correspondence_huber_delta"]),
                    )
                    commitment = details["commitment_loss"]
                    weighted_infocd = infocd * float(
                        training["reconstruction_weight"]
                    )
                    weighted_point_mse = point_mse * float(
                        training["point_mse_weight"]
                    )
                    weighted_huber = huber * float(
                        training["correspondence_huber_weight"]
                    )
                    total_loss = (
                        weighted_infocd
                        + weighted_point_mse
                        + weighted_huber
                        + commitment
                    )
                    displacement_mean, displacement_p95, displacement_max = (
                        _displacement_statistics(details["displacement"])
                    )
                    scalars = np.asarray(
                        jt.stack(
                            (
                                total_loss,
                                infocd,
                                weighted_infocd,
                                point_mse,
                                weighted_point_mse,
                                huber,
                                weighted_huber,
                                commitment,
                            )
                        ).numpy(),
                        dtype=np.float64,
                    ).reshape(-1)
                    if scalars.shape != (8,) or not np.isfinite(scalars).all():
                        raise RuntimeError("training losses are non-finite")
                    gradient_norm = _gradient_norm_and_step(optimizer, total_loss)
                    ema_updates = int(model.apply_pending_ema())
                    row: dict[str, object] = {
                        "epoch": epoch_index + 1,
                        "batch_index": batch_index,
                        "step": global_step,
                        "batch_size": int(batch.pgd1_input.shape[0]),
                        "total_loss": float(scalars[0]),
                        "infocd_loss_raw": float(scalars[1]),
                        "infocd_loss_weighted": float(scalars[2]),
                        "point_mse_raw": float(scalars[3]),
                        "point_mse_weighted": float(scalars[4]),
                        "correspondence_huber_loss_raw": float(scalars[5]),
                        "correspondence_huber_loss_weighted": float(scalars[6]),
                        "commitment_loss": float(scalars[7]),
                        "gradient_norm": gradient_norm,
                        "displacement_mean": displacement_mean,
                        "displacement_p95": displacement_p95,
                        "displacement_max": displacement_max,
                        "learning_rate": learning_rate,
                        "ema_updates": ema_updates,
                    }
                    epoch_rows.append(row)
                    _append_jsonl(step_stream, row)
                    _log_training_row(
                        writer,
                        row,
                        steps_per_epoch=steps_per_epoch,
                        log_every=int(training["log_every"]),
                    )
                    if global_step % int(training["log_every"]) == 0:
                        writer.flush()

                epoch_number = epoch_index + 1
                epoch_summary = summarize_epoch_step_metrics(epoch_rows)
                epoch_summary.update(
                    {
                        "epoch": epoch_number,
                        "global_step": global_step,
                        "elapsed_seconds": time.perf_counter() - epoch_started,
                    }
                )
                _append_jsonl(epoch_stream, epoch_summary)
                step_stream.flush()
                epoch_stream.flush()
                os.fsync(step_stream.fileno())
                os.fsync(epoch_stream.fileno())
                _log_epoch_summary(writer, epoch_summary)

                checkpoint_path = (
                    run_path
                    / "checkpoints"
                    / f"epoch_{epoch_number:04d}_step_{global_step:08d}.pkl"
                )
                checkpoint = save_checkpoint(
                    checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    step=global_step,
                    config_sha256=config_sha256,
                    metrics=epoch_summary,
                    training_state={
                        "completed_epochs": epoch_number,
                        "global_step": global_step,
                        "steps_per_epoch": steps_per_epoch,
                        "sample_ids_sha256": hashlib.sha256(
                            "".join(
                                f"{item}\n" for item in dataset.sample_ids
                            ).encode("ascii")
                        ).hexdigest(),
                        "paired_content_sha256": dataset.content_sha256,
                        "initial_checkpoint_sha256": initial_digest,
                    },
                )
                if (
                    run_validation
                    and epoch_number
                    % int(canonical["evaluation"]["every_epochs"])
                    == 0
                ):
                    _assert_runtime_unchanged(
                        repo_root,
                        expected_sources=source_bindings,
                        expected_runtime=runtime_binding,
                    )
                    last_validation = _run_validation(
                        model=model,
                        config=canonical,
                        config_sha256=config_sha256,
                        checkpoint=checkpoint,
                        epoch_number=epoch_number,
                        run_dir=run_path,
                        repo_root=repo_root,
                    )
                    _assert_runtime_unchanged(
                        repo_root,
                        expected_sources=source_bindings,
                        expected_runtime=runtime_binding,
                    )
                    last_validation["is_best"] = _update_best(
                        run_path, last_validation
                    )
                    _append_jsonl(score_stream, last_validation)
                    score_stream.flush()
                    os.fsync(score_stream.fileno())
                    score_rows.append(last_validation)
                completed_epochs = epoch_number
                _atomic_json(
                    run_path / "run_manifest.json",
                    _run_manifest(
                        status="running",
                        completed_epochs=completed_epochs,
                        global_step=global_step,
                        max_epochs=max_epochs,
                        steps_per_epoch=steps_per_epoch,
                        config_sha256=config_sha256,
                        initial_checkpoint_sha256=initial_digest,
                        last_validation=last_validation,
                    ),
                )
                if last_validation is not None:
                    _log_validation_row(writer, last_validation)
                writer.flush()
        _atomic_json(
            run_path / "run_manifest.json",
            _run_manifest(
                status="completed",
                completed_epochs=completed_epochs,
                global_step=global_step,
                max_epochs=max_epochs,
                steps_per_epoch=steps_per_epoch,
                config_sha256=config_sha256,
                initial_checkpoint_sha256=initial_digest,
                last_validation=last_validation,
            ),
        )
    except BaseException as error:
        _atomic_json(
            run_path / "run_manifest.json",
            _run_manifest(
                status="failed",
                completed_epochs=completed_epochs,
                global_step=global_step,
                max_epochs=max_epochs,
                steps_per_epoch=steps_per_epoch,
                config_sha256=config_sha256,
                initial_checkpoint_sha256=initial_digest,
                last_validation=last_validation,
                error=f"{type(error).__name__}: {error}",
            ),
        )
        raise
    finally:
        writer.close()

    return json.loads((run_path / "run_manifest.json").read_text(encoding="utf-8"))


def run_pgd2_training(
    config: Mapping[str, object],
    *,
    raw_config_sha256: str,
    run_dir: os.PathLike[str] | str,
    initial_checkpoint: os.PathLike[str] | str | None = None,
    expected_initial_checkpoint_sha256: str | None = None,
    resume_checkpoint: os.PathLike[str] | str | None = None,
    run_validation: bool = True,
    command: Sequence[str] | None = None,
) -> dict[str, object]:
    """Run one PGD2 arm while holding an exclusive per-directory lock."""

    run_path = Path(run_dir).resolve()
    with _exclusive_run_lock(run_path):
        return _run_pgd2_training_unlocked(
            config,
            raw_config_sha256=raw_config_sha256,
            run_dir=run_path,
            initial_checkpoint=initial_checkpoint,
            expected_initial_checkpoint_sha256=expected_initial_checkpoint_sha256,
            resume_checkpoint=resume_checkpoint,
            run_validation=run_validation,
            command=command,
        )


__all__ = [
    "PGD2_CONFIG_FORMAT",
    "PGD2_SAMPLER",
    "load_pgd2_config",
    "pgd2_model_config_sha256",
    "run_pgd2_training",
    "summarize_epoch_step_metrics",
    "validate_pgd2_training_config",
]

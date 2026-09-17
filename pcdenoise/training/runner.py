"""Auditable pure-Jittor PGD training loop."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from numbers import Integral, Real
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, TypeVar

import jittor as jt
import numpy as np

from pcdenoise.data.noise import NOISE_PROFILES, ROTATION_MODES
from pcdenoise.data.patch_center_plan import (
    PATCH_CENTER_MODE,
    PatchCenterPlan,
    verify_patch_center_plan,
    verify_patch_center_source_cache,
)
from pcdenoise.data.pgd_training import TRAINING_NORMALIZATION_MODES
from pcdenoise.models.factory import (
    MODEL_ARCHITECTURES,
    build_pgd_model,
)
from pcdenoise.models.losses import (
    INFOCD_PROFILES,
    correspondence_huber_loss,
    infocd_loss,
)
from pcdenoise.ops.chamfer import chamfer_distance
from pcdenoise.training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from pcdenoise.training.engine import (
    SURFACE_VIEW_MODE,
    build_epoch_training_batch,
    build_training_batch,
    canonical_config_sha256,
    scan_clean_cache,
)
from pcdenoise.training.logger import TensorBoardLogger, scalar_to_float
from pcdenoise.training.manifest import create_run
from pcdenoise.training.sampling import (
    SAMPLER_VERSION,
    coverage_statistics,
    epoch_batches,
)


TRAIN_CONFIG_FORMAT = "pcdenoise_pgd_train_v1"
EPOCH_TRAIN_CONFIG_FORMAT = "pcdenoise_pgd_train_v2"
_LEARNING_RATE_SCHEDULES = ("constant", "warmup_cosine_v1")
_PREFETCH_THREAD_NAME_PREFIX = "pcdenoise-batch-prefetch"
_NOISE_SCALE_DIAGNOSTIC_LIMIT = 5
_BatchResult = TypeVar("_BatchResult")


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _keys(
    values: Mapping[str, object],
    *,
    name: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    missing = sorted(required - set(values))
    extra = sorted(set(values) - allowed)
    if missing:
        raise ValueError(f"{name} is missing keys: {missing}")
    if extra:
        raise ValueError(f"{name} contains unknown keys: {extra}")


def _integer(
    value: object,
    *,
    name: str,
    minimum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < minimum
    ):
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return int(value)


def _real(
    value: object,
    *,
    name: str,
    minimum: float,
    strictly_greater: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    invalid_bound = (
        converted <= minimum
        if strictly_greater
        else converted < minimum
    )
    if not math.isfinite(converted) or invalid_bound:
        comparator = "greater than" if strictly_greater else "at least"
        raise ValueError(f"{name} must be finite and {comparator} {minimum}")
    return converted


def _bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


@contextmanager
def _ordered_prefetch_batches(
    batch_indices: Iterable[int],
    build_batch: Callable[[int], _BatchResult],
    *,
    prefetch_batches: int,
) -> Iterator[Iterator[tuple[int, _BatchResult]]]:
    """Yield fully built batches in input order with one bounded CPU worker."""

    depth = _integer(
        prefetch_batches,
        name="prefetch_batches",
        minimum=0,
    )
    indices = iter(batch_indices)
    if depth == 0:
        def synchronous() -> Iterator[tuple[int, _BatchResult]]:
            for batch_index in indices:
                yield batch_index, build_batch(batch_index)

        yield synchronous()
        return

    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix=_PREFETCH_THREAD_NAME_PREFIX,
    )
    pending: deque[tuple[int, Future[_BatchResult]]] = deque()

    def submit_next() -> bool:
        try:
            batch_index = next(indices)
        except StopIteration:
            return False
        pending.append(
            (batch_index, executor.submit(build_batch, batch_index))
        )
        return True

    try:
        for _ in range(depth):
            if not submit_next():
                break

        def prefetched() -> Iterator[tuple[int, _BatchResult]]:
            while pending:
                batch_index, future = pending.popleft()
                batch = future.result()
                submit_next()
                yield batch_index, batch

        yield prefetched()
    finally:
        while pending:
            _batch_index, future = pending.popleft()
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _integer_list(
    value: object,
    *,
    name: str,
    length: int,
) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} integers")
    return [
        _integer(item, name=f"{name}[{index}]", minimum=1)
        for index, item in enumerate(value)
    ]


def _validated_learning_rate_schedule(
    value: object,
    *,
    learning_rate: float,
) -> str | dict[str, object]:
    if isinstance(value, str):
        if value not in _LEARNING_RATE_SCHEDULES:
            raise ValueError(
                "learning_rate_schedule must be 'constant' or "
                "'warmup_cosine_v1'"
            )
        return value
    schedule = _mapping(value, name="learning_rate_schedule")
    _keys(
        schedule,
        name="learning_rate_schedule",
        required={"name"},
        optional={"warmup_steps", "min_learning_rate"},
    )
    name = schedule["name"]
    if name not in _LEARNING_RATE_SCHEDULES:
        raise ValueError(
            "learning_rate_schedule.name must be 'constant' or "
            "'warmup_cosine_v1'"
        )
    if name == "constant":
        if set(schedule) != {"name"}:
            raise ValueError(
                "constant learning_rate_schedule accepts only name"
            )
        return {"name": "constant"}
    canonical: dict[str, object] = {"name": "warmup_cosine_v1"}
    if "warmup_steps" in schedule:
        canonical["warmup_steps"] = _integer(
            schedule["warmup_steps"],
            name="learning_rate_schedule.warmup_steps",
            minimum=1,
        )
    if "min_learning_rate" in schedule:
        minimum = _real(
            schedule["min_learning_rate"],
            name="learning_rate_schedule.min_learning_rate",
            minimum=0.0,
            strictly_greater=False,
        )
        if minimum > learning_rate:
            raise ValueError(
                "learning_rate_schedule.min_learning_rate must not "
                "exceed learning_rate"
            )
        canonical["min_learning_rate"] = minimum
    return canonical


def _validated_config(config: Mapping[str, object]) -> dict[str, object]:
    source = _mapping(config, name="config")
    _keys(
        source,
        name="config",
        required={"format", "model", "data", "training"},
    )
    config_format = source["format"]
    if config_format not in (
        TRAIN_CONFIG_FORMAT,
        EPOCH_TRAIN_CONFIG_FORMAT,
    ):
        raise ValueError(
            f"config format must be {TRAIN_CONFIG_FORMAT!r} or "
            f"{EPOCH_TRAIN_CONFIG_FORMAT!r}"
        )
    is_epoch_config = config_format == EPOCH_TRAIN_CONFIG_FORMAT

    model = _mapping(source["model"], name="model")
    _keys(
        model,
        name="model",
        required={
            "patch_size",
            "feature_dims",
            "codebook_sizes",
            "temperature",
            "ema_momentum",
            "displacement_scale",
        },
        optional=(
            {
                "architecture",
                "noise_conditioning",
                "pointwise_displacement_gate",
                "residual_refinement",
                "residual_refinement_hidden_dim",
                "residual_refinement_scale",
            }
            if is_epoch_config
            else set()
        ),
    )
    canonical_model = {
        "patch_size": _integer(
            model["patch_size"], name="patch_size", minimum=1
        ),
        "feature_dims": _integer_list(
            model["feature_dims"], name="feature_dims", length=5
        ),
        "codebook_sizes": _integer_list(
            model["codebook_sizes"], name="codebook_sizes", length=4
        ),
        "temperature": _real(
            model["temperature"],
            name="temperature",
            minimum=0.0,
            strictly_greater=True,
        ),
        "ema_momentum": _real(
            model["ema_momentum"],
            name="ema_momentum",
            minimum=0.0,
            strictly_greater=True,
        ),
        "displacement_scale": _real(
            model["displacement_scale"],
            name="displacement_scale",
            minimum=0.0,
            strictly_greater=True,
        ),
    }
    if canonical_model["ema_momentum"] >= 1.0:
        raise ValueError("ema_momentum must be smaller than 1")
    architecture = model.get("architecture", "pgd")
    if (
        not isinstance(architecture, str)
        or architecture not in MODEL_ARCHITECTURES
    ):
        raise ValueError(
            "model.architecture must be one of "
            f"{MODEL_ARCHITECTURES}"
        )
    if "architecture" in model:
        canonical_model["architecture"] = architecture
    for optional_flag in (
        "noise_conditioning",
        "pointwise_displacement_gate",
        "residual_refinement",
    ):
        if optional_flag in model:
            canonical_model[optional_flag] = _bool(
                model[optional_flag],
                name=f"model.{optional_flag}",
            )
    refinement_enabled = bool(
        canonical_model.get("residual_refinement", False)
    )
    refinement_parameters = {
        "residual_refinement_hidden_dim",
        "residual_refinement_scale",
    }
    if refinement_enabled:
        if "residual_refinement_hidden_dim" in model:
            canonical_model["residual_refinement_hidden_dim"] = _integer(
                model["residual_refinement_hidden_dim"],
                name="model.residual_refinement_hidden_dim",
                minimum=1,
            )
        if "residual_refinement_scale" in model:
            canonical_model["residual_refinement_scale"] = _real(
                model["residual_refinement_scale"],
                name="model.residual_refinement_scale",
                minimum=0.0,
                strictly_greater=True,
            )
    elif refinement_parameters & set(model):
        raise ValueError(
            "model.residual_refinement_hidden_dim/scale requires "
            "model.residual_refinement=true"
        )
    data = _mapping(source["data"], name="data")
    optional_verification = {
        "mesh_root",
        "train_split",
        "expected_train_split_sha256",
        "expected_split_manifest_sha256",
        "expected_content_sha256",
    }
    optional_verification_mode = {"verify_files"}
    optional_augmentation = (
        {
            "rotation_mode",
            "normalization_mode",
            "conditioning_noise_min",
            "conditioning_noise_max",
            "patch_center",
            "surface_view",
        }
        if is_epoch_config
        else set()
    )
    _keys(
        data,
        name="data",
        required={
            "train_cache",
            "verify_cache",
            "noise_min",
            "noise_max",
            "rotate",
        }
        | ({"noise_profile"} if is_epoch_config else set()),
        optional=(
            optional_verification
            | optional_verification_mode
            | optional_augmentation
        ),
    )
    cache_root = Path(str(data["train_cache"])).resolve()
    if not cache_root.is_dir():
        raise FileNotFoundError(cache_root)
    noise_min = _real(
        data["noise_min"],
        name="noise_min",
        minimum=0.0,
        strictly_greater=False,
    )
    noise_max = _real(
        data["noise_max"],
        name="noise_max",
        minimum=noise_min,
        strictly_greater=True,
    )
    verify_cache = _bool(data["verify_cache"], name="verify_cache")
    canonical_data: dict[str, object] = {
        "train_cache": str(cache_root),
        "verify_cache": verify_cache,
        "noise_min": noise_min,
        "noise_max": noise_max,
        "rotate": _bool(data["rotate"], name="rotate"),
    }
    if "verify_files" in data:
        canonical_data["verify_files"] = _bool(
            data["verify_files"],
            name="verify_files",
        )
    if is_epoch_config:
        noise_profile = data["noise_profile"]
        if noise_profile not in NOISE_PROFILES:
            raise ValueError(
                f"noise_profile must be one of {NOISE_PROFILES}"
            )
        canonical_data["noise_profile"] = noise_profile
        if "patch_center" in data:
            patch_center = _mapping(
                data["patch_center"],
                name="data.patch_center",
            )
            _keys(
                patch_center,
                name="data.patch_center",
                required={
                    "mode",
                    "plan_dir",
                    "expected_content_sha256",
                    "epoch_offset",
                },
            )
            if patch_center["mode"] != PATCH_CENTER_MODE:
                raise ValueError(
                    f"data.patch_center.mode must be {PATCH_CENTER_MODE!r}"
                )
            plan_dir = Path(str(patch_center["plan_dir"])).resolve()
            if not plan_dir.is_dir():
                raise FileNotFoundError(plan_dir)
            canonical_data["patch_center"] = {
                "mode": PATCH_CENTER_MODE,
                "plan_dir": str(plan_dir),
                "expected_content_sha256": _sha256(
                    patch_center["expected_content_sha256"],
                    name="data.patch_center.expected_content_sha256",
                ),
                "epoch_offset": _integer(
                    patch_center["epoch_offset"],
                    name="data.patch_center.epoch_offset",
                    minimum=0,
                ),
            }
        if "surface_view" in data:
            surface_view = _mapping(
                data["surface_view"],
                name="data.surface_view",
            )
            _keys(
                surface_view,
                name="data.surface_view",
                required={"mode", "view_epoch_offset"},
            )
            if surface_view["mode"] != SURFACE_VIEW_MODE:
                raise ValueError(
                    "data.surface_view.mode must be "
                    f"{SURFACE_VIEW_MODE!r}"
                )
            canonical_data["surface_view"] = {
                "mode": SURFACE_VIEW_MODE,
                "view_epoch_offset": _integer(
                    surface_view["view_epoch_offset"],
                    name="data.surface_view.view_epoch_offset",
                    minimum=0,
                ),
            }
        if "rotation_mode" in data:
            rotation_mode = data["rotation_mode"]
            if rotation_mode not in ROTATION_MODES:
                raise ValueError(
                    f"rotation_mode must be one of {ROTATION_MODES}"
                )
            canonical_data["rotation_mode"] = rotation_mode
        if "normalization_mode" in data:
            normalization_mode = data["normalization_mode"]
            if normalization_mode not in TRAINING_NORMALIZATION_MODES:
                raise ValueError(
                    "normalization_mode must be one of "
                    f"{TRAINING_NORMALIZATION_MODES}"
                )
            canonical_data["normalization_mode"] = normalization_mode
        conditioning_keys = {
            "conditioning_noise_min",
            "conditioning_noise_max",
        }
        present_conditioning_keys = conditioning_keys & set(data)
        if present_conditioning_keys and (
            present_conditioning_keys != conditioning_keys
        ):
            raise ValueError(
                "conditioning_noise_min and conditioning_noise_max "
                "must be supplied together"
            )
        if present_conditioning_keys:
            conditioning_minimum = _real(
                data["conditioning_noise_min"],
                name="conditioning_noise_min",
                minimum=0.0,
                strictly_greater=True,
            )
            conditioning_maximum = _real(
                data["conditioning_noise_max"],
                name="conditioning_noise_max",
                minimum=conditioning_minimum,
                strictly_greater=True,
            )
            canonical_data["conditioning_noise_min"] = (
                conditioning_minimum
            )
            canonical_data["conditioning_noise_max"] = (
                conditioning_maximum
            )
    if (
        bool(canonical_model.get("noise_conditioning", False))
        and noise_min <= 0.0
    ):
        raise ValueError(
            "data.noise_min must be greater than 0.0 when "
            "model.noise_conditioning=true"
        )
    conditioning_enabled = bool(
        canonical_model.get("noise_conditioning", False)
    )
    has_conditioning_range = (
        "conditioning_noise_min" in canonical_data
    )
    if has_conditioning_range and not conditioning_enabled:
        raise ValueError(
            "conditioning_noise_min/conditioning_noise_max require "
            "model.noise_conditioning=true"
        )
    actual_normalization_mode = str(
        canonical_data.get("normalization_mode", "clean_unit")
    )
    if (
        is_epoch_config
        and conditioning_enabled
        and actual_normalization_mode == "noisy_max"
        and not has_conditioning_range
    ):
        raise ValueError(
            "noisy_max with model.noise_conditioning=true requires "
            "explicit conditioning_noise_min and conditioning_noise_max"
        )
    if (
        has_conditioning_range
        and actual_normalization_mode == "clean_unit"
        and (
            float(canonical_data["conditioning_noise_min"]) > noise_min
            or float(canonical_data["conditioning_noise_max"]) < noise_max
        )
    ):
        raise ValueError(
            "conditioning noise range must contain the sampled noise range "
            "when normalization_mode='clean_unit'"
        )
    if verify_cache:
        missing = sorted(optional_verification - set(data))
        if missing:
            raise ValueError(
                f"verified data config is missing keys: {missing}"
            )
        for path_key in ("mesh_root", "train_split"):
            path = Path(str(data[path_key])).resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            canonical_data[path_key] = str(path)
        for digest_key in (
            "expected_train_split_sha256",
            "expected_split_manifest_sha256",
            "expected_content_sha256",
        ):
            canonical_data[digest_key] = _sha256(
                data[digest_key],
                name=digest_key,
            )
    elif optional_verification & set(data):
        raise ValueError(
            "verification paths/hashes require verify_cache=true"
        )
    elif optional_verification_mode & set(data):
        raise ValueError("verify_files requires verify_cache=true")

    training = _mapping(source["training"], name="training")
    lifecycle_keys = (
        {"sampler", "max_epochs", "learning_rate_schedule"}
        if is_epoch_config
        else {"max_steps"}
    )
    _keys(
        training,
        name="training",
        required={
            "seed",
            "batch_size",
            "learning_rate",
            "loss_profile",
            "reconstruction_weight",
            "point_mse_weight",
            "log_every",
            "visualize_every",
            "checkpoint_every",
            "use_cuda",
        }
        | lifecycle_keys,
        optional={
            "paper_alpha",
            "squared_chamfer_weight",
            "target_coverage_weight",
            "relative_squared_chamfer_weight",
            "trainable_scope",
        }
        | (
            {
                "prefetch_batches",
                "noise_scale_estimation_weight",
                "correspondence_huber_weight",
                "correspondence_huber_delta",
            }
            if is_epoch_config
            else set()
        ),
    )
    seed = _integer(training["seed"], name="seed", minimum=0)
    if seed >= 2**64:
        raise ValueError("seed must be smaller than 2**64")
    profile = training["loss_profile"]
    if profile not in INFOCD_PROFILES:
        raise ValueError(f"loss_profile must be one of {INFOCD_PROFILES}")
    reconstruction_weight = _real(
        training["reconstruction_weight"],
        name="reconstruction_weight",
        minimum=0.0,
        strictly_greater=False,
    )
    point_mse_weight = _real(
        training["point_mse_weight"],
        name="point_mse_weight",
        minimum=0.0,
        strictly_greater=False,
    )
    learning_rate = _real(
        training["learning_rate"],
        name="learning_rate",
        minimum=0.0,
        strictly_greater=True,
    )
    canonical_training: dict[str, object] = {
        "seed": seed,
        "batch_size": _integer(
            training["batch_size"], name="batch_size", minimum=1
        ),
        "learning_rate": learning_rate,
        "loss_profile": profile,
        "reconstruction_weight": reconstruction_weight,
        "point_mse_weight": point_mse_weight,
        "log_every": _integer(
            training["log_every"], name="log_every", minimum=1
        ),
        "visualize_every": _integer(
            training["visualize_every"],
            name="visualize_every",
            minimum=1,
        ),
        "checkpoint_every": _integer(
            training["checkpoint_every"],
            name="checkpoint_every",
            minimum=1,
        ),
        "use_cuda": _bool(training["use_cuda"], name="use_cuda"),
    }
    if "trainable_scope" in training:
        trainable_scope = training["trainable_scope"]
        if trainable_scope not in ("all", "residual_refinement"):
            raise ValueError(
                "training.trainable_scope must be 'all' or "
                "'residual_refinement'"
            )
        if trainable_scope == "residual_refinement":
            if not is_epoch_config:
                raise ValueError(
                    "training.trainable_scope='residual_refinement' "
                    "requires an epoch v2 config"
                )
            if not refinement_enabled:
                raise ValueError(
                    "training.trainable_scope='residual_refinement' "
                    "requires model.residual_refinement=true"
                )
        canonical_training["trainable_scope"] = trainable_scope
    if is_epoch_config:
        if training["sampler"] != SAMPLER_VERSION:
            raise ValueError(
                f"sampler must be {SAMPLER_VERSION!r}"
            )
        canonical_training.update(
            {
                "sampler": SAMPLER_VERSION,
                "max_epochs": _integer(
                    training["max_epochs"],
                    name="max_epochs",
                    minimum=1,
                ),
                "learning_rate_schedule": (
                    _validated_learning_rate_schedule(
                        training["learning_rate_schedule"],
                        learning_rate=learning_rate,
                    )
                ),
            }
        )
        if "prefetch_batches" in training:
            prefetch_batches = _integer(
                training["prefetch_batches"],
                name="prefetch_batches",
                minimum=0,
            )
            if prefetch_batches > 0:
                canonical_training["prefetch_batches"] = prefetch_batches
    else:
        canonical_training["max_steps"] = _integer(
            training["max_steps"], name="max_steps", minimum=1
        )
    for optional_weight in (
        "squared_chamfer_weight",
        "target_coverage_weight",
        "relative_squared_chamfer_weight",
    ):
        if optional_weight in training:
            canonical_training[optional_weight] = _real(
                training[optional_weight],
                name=optional_weight,
                minimum=0.0,
                strictly_greater=False,
            )
    conditioning_enabled = bool(
        canonical_model.get("noise_conditioning", False)
    )
    if conditioning_enabled:
        if "noise_scale_estimation_weight" not in training:
            raise ValueError(
                "noise_scale_estimation_weight is required when "
                "model.noise_conditioning=true"
            )
        canonical_training["noise_scale_estimation_weight"] = _real(
            training["noise_scale_estimation_weight"],
            name="noise_scale_estimation_weight",
            minimum=0.0,
            strictly_greater=True,
        )
    elif "noise_scale_estimation_weight" in training:
        raise ValueError(
            "noise_scale_estimation_weight requires "
            "model.noise_conditioning=true"
        )
    huber_weight = 0.0
    if "correspondence_huber_weight" in training:
        huber_weight = _real(
            training["correspondence_huber_weight"],
            name="correspondence_huber_weight",
            minimum=0.0,
            strictly_greater=False,
        )
    if huber_weight > 0.0:
        canonical_training["correspondence_huber_weight"] = huber_weight
        canonical_training["correspondence_huber_delta"] = _real(
            training.get("correspondence_huber_delta", 0.005),
            name="correspondence_huber_delta",
            minimum=0.0,
            strictly_greater=True,
        )
    elif "correspondence_huber_delta" in training:
        raise ValueError(
            "correspondence_huber_delta requires a positive "
            "correspondence_huber_weight"
        )
    if not any(
        float(canonical_training.get(weight, 0.0)) > 0.0
        for weight in (
            "reconstruction_weight",
            "point_mse_weight",
            "squared_chamfer_weight",
            "target_coverage_weight",
            "relative_squared_chamfer_weight",
            "correspondence_huber_weight",
        )
    ):
        raise ValueError(
            "at least one reconstruction loss weight must be positive"
        )
    if profile == "paper":
        if "paper_alpha" not in training:
            raise ValueError("paper_alpha is required for the paper loss")
        canonical_training["paper_alpha"] = _real(
            training["paper_alpha"],
            name="paper_alpha",
            minimum=0.0,
            strictly_greater=True,
        )
    elif "paper_alpha" in training:
        raise ValueError(
            f"paper_alpha must be omitted for {profile} loss"
        )
    return {
        "format": config_format,
        "model": canonical_model,
        "data": canonical_data,
        "training": canonical_training,
    }


def validate_training_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Return the canonical, type-exact training configuration."""

    return _validated_config(config)


def _optional_chamfer_terms(
    prediction: jt.Var,
    target: jt.Var,
    training_config: Mapping[str, object],
) -> tuple[jt.Var, jt.Var] | None:
    squared_weight = float(
        training_config.get("squared_chamfer_weight", 0.0)
    )
    coverage_weight = float(
        training_config.get("target_coverage_weight", 0.0)
    )
    if squared_weight == 0.0 and coverage_weight == 0.0:
        return None
    squared_chamfer, _, target_to_prediction = chamfer_distance(
        prediction,
        target,
        return_components=True,
        validate_finite=False,
    )
    return squared_chamfer, target_to_prediction.mean()


def _optional_relative_squared_chamfer(
    prediction: jt.Var,
    noisy: jt.Var,
    target: jt.Var,
    training_config: Mapping[str, object],
) -> jt.Var | None:
    weight = float(
        training_config.get("relative_squared_chamfer_weight", 0.0)
    )
    if weight == 0.0:
        return None
    _, prediction_to_target, target_to_prediction = chamfer_distance(
        prediction,
        target,
        return_components=True,
        validate_finite=False,
    )
    _, noisy_to_target, target_to_noisy = chamfer_distance(
        noisy,
        target,
        return_components=True,
        validate_finite=False,
    )
    denominator = jt.maximum(
        (noisy_to_target + target_to_noisy).stop_grad(),
        1e-12,
    )
    return (
        (prediction_to_target + target_to_prediction) / denominator
    ).mean()


def _epoch_model_forward(
    model: object,
    noisy: jt.Var,
    *,
    noise_scales: object,
    model_config: Mapping[str, object],
) -> tuple[jt.Var, Mapping[str, object], jt.Var | None]:
    """Run one v2 forward pass and retain scales only as auxiliary targets."""

    if bool(model_config.get("noise_conditioning", False)):
        teacher = jt.array(noise_scales).float32().reshape((-1, 1))
        prediction, details = model(
            noisy,
            return_details=True,
        )
        return prediction, details, teacher
    prediction, details = model(noisy, return_details=True)
    return prediction, details, None


def _validate_effective_noise_scales(
    noise_scales: object,
    *,
    data_config: Mapping[str, object],
    epoch: int | None = None,
    batch_index: int | None = None,
    global_step: int | None = None,
    sample_ids: tuple[str, ...] | None = None,
    source_noise_scales: object | None = None,
    normalization_scales: object | None = None,
    center_indices: tuple[int, ...] | None = None,
    view_ids: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Fail closed with at most five batch records for range diagnosis."""

    values = np.asarray(noise_scales)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
        or float(values.min()) <= 0.0
    ):
        raise RuntimeError("effective noise-scale teachers are invalid")
    minimum = np.float32(
        data_config.get(
            "conditioning_noise_min",
            data_config["noise_min"],
        )
    )
    maximum = np.float32(
        data_config.get(
            "conditioning_noise_max",
            data_config["noise_max"],
        )
    )
    actual_minimum = float(values.min())
    actual_maximum = float(values.max())
    below = values < minimum
    above = values > maximum
    if bool(below.any()) or bool(above.any()):
        offending_indices = np.flatnonzero(np.logical_or(below, above))
        shown_indices = offending_indices[:_NOISE_SCALE_DIAGNOSTIC_LIMIT]
        source_values = (
            np.asarray(source_noise_scales)
            if source_noise_scales is not None
            else None
        )
        normalization_values = (
            np.asarray(normalization_scales)
            if normalization_scales is not None
            else None
        )

        def available(value: object | None, index: int) -> bool:
            if value is None:
                return False
            try:
                size = len(value)  # type: ignore[arg-type]
                return size > index
            except TypeError:
                return False

        offenders = []
        for raw_index in shown_indices:
            index = int(raw_index)
            fields = [
                f"index={index}",
                f"effective={float(values[index]):.7g}",
            ]
            if available(sample_ids, index):
                fields.insert(1, f"sample_id={sample_ids[index]}")
            if available(source_values, index):
                fields.append(
                    f"source={float(source_values[index]):.7g}"
                )
            if available(normalization_values, index):
                fields.append(
                    "normalization_scale="
                    f"{float(normalization_values[index]):.7g}"
                )
            if available(center_indices, index):
                fields.append(f"center_index={int(center_indices[index])}")
            if available(view_ids, index):
                fields.append(f"view_id={int(view_ids[index])}")
            offenders.append("{" + ",".join(fields) + "}")

        context = []
        for name, value in (
            ("epoch", epoch),
            ("batch_index", batch_index),
            ("global_step", global_step),
        ):
            if value is not None:
                context.append(f"{name}={int(value)}")
        offender_count = int(len(offending_indices))
        shown_count = int(len(shown_indices))
        raise RuntimeError(
            "effective noise-scale teacher is outside the configured "
            "conditioning range "
            f"[{float(minimum):.7g}, {float(maximum):.7g}]; "
            f"actual_min={actual_minimum:.7g}; "
            f"actual_max={actual_maximum:.7g}; "
            f"below_count={int(np.count_nonzero(below))}; "
            f"above_count={int(np.count_nonzero(above))}; "
            + ("; ".join(context) + "; " if context else "")
            + f"offender_count={offender_count}; "
            f"shown_count={shown_count}; "
            f"omitted_count={offender_count - shown_count}; "
            f"offenders=[{','.join(offenders)}]; "
            "refusing to clip"
        )
    return values


def _conditioned_scale_terms(
    details: Mapping[str, object],
    teacher: jt.Var | None,
) -> tuple[jt.Var, jt.Var, jt.Var] | None:
    """Return log-scale MSE, estimate, and signed error when enabled."""

    if teacher is None:
        return None
    estimated = details.get("estimated_noise_scale")
    if not isinstance(estimated, jt.Var):
        raise RuntimeError(
            "conditioned model did not return estimated_noise_scale"
        )
    if tuple(estimated.shape) != tuple(teacher.shape):
        raise RuntimeError(
            "estimated_noise_scale shape does not match the teacher"
        )
    error = estimated - teacher
    loss = ((jt.log(estimated) - jt.log(teacher)) ** 2).mean()
    return loss, estimated, error


def _displacement_gate(
    details: Mapping[str, object],
    *,
    enabled: bool,
) -> jt.Var | None:
    if not enabled:
        return None
    gate = details.get("displacement_gate")
    if not isinstance(gate, jt.Var):
        raise RuntimeError(
            "gated model did not return displacement_gate"
        )
    return gate


def _cache_manifest(
    cache_root: Path,
    samples: list[object],
) -> dict[str, object]:
    manifest_path = cache_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid surface-cache manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise ValueError("surface-cache manifest must be a mapping")
    raw_ids = manifest.get("shape_ids")
    count = manifest.get("shape_count")
    sample_ids = [sample.sample_id for sample in samples]
    if (
        not isinstance(raw_ids, list)
        or raw_ids != sample_ids
        or isinstance(count, bool)
        or count != len(sample_ids)
    ):
        raise ValueError(
            "surface-cache manifest shape IDs/count do not match scanned shapes"
        )
    _sha256(manifest.get("split_sha256"), name="cache split_sha256")
    _sha256(manifest.get("content_sha256"), name="cache content_sha256")
    if manifest.get("format") == "pcdenoise_train_surface_cache_v2":
        if manifest.get("format_version") != 2:
            raise ValueError("surface-cache v2 version is invalid")
        view_count = _integer(
            manifest.get("view_count"),
            name="cache view_count",
            minimum=1,
        )
        if view_count not in (1, 2):
            raise ValueError("cache view_count must be 1 or 2")
        records = manifest.get("samples")
        if not isinstance(records, list) or len(records) != len(samples):
            raise ValueError("surface-cache v2 sample records are invalid")
        for sample, record in zip(samples, records):
            if (
                sample.view_count != view_count
                or not isinstance(record, Mapping)
                or record.get("shape_id") != sample.sample_id
            ):
                raise ValueError("surface-cache v2 views do not match scan")
            views = record.get("views")
            if not isinstance(views, list) or len(views) != view_count:
                raise ValueError("surface-cache v2 view records are invalid")
            expected_paths = tuple(
                str(path.resolve()) for path in sample.all_view_paths
            )
            declared_paths = tuple(
                str((cache_root / str(view.get("relative_path"))).resolve())
                if isinstance(view, Mapping)
                else ""
                for view in views
            )
            if declared_paths != expected_paths:
                raise ValueError("surface-cache v2 view paths do not match")
    return manifest


def _verify_formal_cache(
    config: Mapping[str, object],
) -> None:
    data = config["data"]
    if not data["verify_cache"]:
        return
    from pcdenoise.data.surface_cache import verify_surface_cache

    verify_files = bool(data.get("verify_files", True))
    verify_surface_cache(
        Path(data["train_cache"]),
        mesh_root=(Path(data["mesh_root"]) if verify_files else None),
        train_split=Path(data["train_split"]),
        expected_train_split_sha256=data[
            "expected_train_split_sha256"
        ],
        expected_split_manifest_sha256=data[
            "expected_split_manifest_sha256"
        ],
        expected_content_sha256=data["expected_content_sha256"],
        verify_files=verify_files,
    )


def _sample_ids_sha256(sample_ids: object) -> str:
    if not isinstance(sample_ids, (list, tuple)) or not sample_ids:
        raise ValueError("sample_ids must be a nonempty list or tuple")
    if any(
        not isinstance(sample_id, str) or not sample_id
        for sample_id in sample_ids
    ):
        raise ValueError("sample_ids must contain nonempty strings")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample_ids must not contain duplicates")
    return hashlib.sha256(
        ("".join(f"{sample_id}\n" for sample_id in sample_ids)).encode(
            "ascii"
        )
    ).hexdigest()


def _learning_rate_at_step(
    training_config: Mapping[str, object],
    *,
    step: int,
    max_steps: int,
    steps_per_epoch: int,
) -> float:
    actual_step = _integer(step, name="step", minimum=1)
    total_steps = _integer(max_steps, name="max_steps", minimum=1)
    epoch_steps = _integer(
        steps_per_epoch,
        name="steps_per_epoch",
        minimum=1,
    )
    if actual_step > total_steps:
        raise ValueError("step must not exceed max_steps")
    base = _real(
        training_config.get("learning_rate"),
        name="learning_rate",
        minimum=0.0,
        strictly_greater=True,
    )
    schedule = training_config.get("learning_rate_schedule")
    if isinstance(schedule, Mapping):
        name = schedule.get("name")
        warmup_steps = schedule.get("warmup_steps", epoch_steps)
        minimum = schedule.get("min_learning_rate", 0.0)
    else:
        name = schedule
        warmup_steps = epoch_steps
        minimum = 0.0
    if name == "constant":
        return base
    if name != "warmup_cosine_v1":
        raise ValueError("learning_rate_schedule is invalid")
    warmup = _integer(
        warmup_steps,
        name="learning_rate_schedule.warmup_steps",
        minimum=1,
    )
    minimum_rate = _real(
        minimum,
        name="learning_rate_schedule.min_learning_rate",
        minimum=0.0,
        strictly_greater=False,
    )
    if minimum_rate > base:
        raise ValueError("min_learning_rate must not exceed learning_rate")
    if actual_step <= warmup:
        return base * (actual_step / warmup)
    if total_steps <= warmup:
        return base
    progress = (actual_step - warmup) / (total_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_rate + (base - minimum_rate) * cosine


def _set_optimizer_learning_rate(
    optimizer: object,
    learning_rate: float,
) -> None:
    actual = _real(
        learning_rate,
        name="actual learning rate",
        minimum=0.0,
        strictly_greater=False,
    )
    if not hasattr(optimizer, "lr"):
        raise TypeError("optimizer must expose lr")
    optimizer.lr = actual
    groups = getattr(optimizer, "param_groups", None)
    if not isinstance(groups, list):
        raise TypeError("optimizer must expose param_groups")
    for group in groups:
        if not isinstance(group, dict):
            raise TypeError("optimizer param_groups must be mappings")
        if "lr" in group:
            group["lr"] = actual


def _validated_epoch_training_state(
    state: object,
    *,
    checkpoint_step: int,
    sample_ids: object,
    steps_per_epoch: int,
    batch_size: int,
    max_epochs: int,
) -> dict[str, object]:
    values = _mapping(state, name="training_state")
    required = {
        "sampler_version",
        "global_step",
        "next_epoch",
        "next_batch_index",
        "sample_count",
        "sample_ids_sha256",
        "steps_per_epoch",
        "batch_size",
        "completed_epochs",
        "current_epoch_coverage",
    }
    _keys(
        values,
        name="training_state",
        required=required,
    )
    if values["sampler_version"] != SAMPLER_VERSION:
        raise ValueError(
            f"training_state.sampler_version must be {SAMPLER_VERSION!r}"
        )
    expected_ids = tuple(sample_ids)
    expected_count = len(expected_ids)
    expected_ids_sha256 = _sample_ids_sha256(expected_ids)
    expected_steps = _integer(
        steps_per_epoch,
        name="steps_per_epoch",
        minimum=1,
    )
    expected_batch_size = _integer(
        batch_size,
        name="batch_size",
        minimum=1,
    )
    epoch_limit = _integer(
        max_epochs,
        name="max_epochs",
        minimum=1,
    )
    global_step = _integer(
        values["global_step"],
        name="training_state.global_step",
        minimum=0,
    )
    loaded_step = _integer(
        checkpoint_step,
        name="checkpoint_step",
        minimum=0,
    )
    if global_step != loaded_step:
        raise ValueError(
            "training_state.global_step does not match checkpoint step"
        )
    next_epoch = _integer(
        values["next_epoch"],
        name="training_state.next_epoch",
        minimum=0,
    )
    next_batch = _integer(
        values["next_batch_index"],
        name="training_state.next_batch_index",
        minimum=0,
    )
    sample_count = _integer(
        values["sample_count"],
        name="training_state.sample_count",
        minimum=1,
    )
    if sample_count != expected_count:
        raise ValueError(
            "training_state.sample_count does not match the cache"
        )
    sample_digest = _sha256(
        values["sample_ids_sha256"],
        name="training_state.sample_ids_sha256",
    )
    if sample_digest != expected_ids_sha256:
        raise ValueError(
            "training_state.sample_ids_sha256 does not match the cache"
        )
    state_steps = _integer(
        values["steps_per_epoch"],
        name="training_state.steps_per_epoch",
        minimum=1,
    )
    if state_steps != expected_steps:
        raise ValueError(
            "training_state.steps_per_epoch does not match the run"
        )
    state_batch_size = _integer(
        values["batch_size"],
        name="training_state.batch_size",
        minimum=1,
    )
    if state_batch_size != expected_batch_size:
        raise ValueError(
            "training_state.batch_size does not match the run"
        )
    completed_epochs = _integer(
        values["completed_epochs"],
        name="training_state.completed_epochs",
        minimum=0,
    )
    coverage = _real(
        values["current_epoch_coverage"],
        name="training_state.current_epoch_coverage",
        minimum=0.0,
        strictly_greater=False,
    )
    if coverage > 1.0:
        raise ValueError(
            "training_state.current_epoch_coverage must not exceed 1"
        )
    if next_epoch > epoch_limit:
        raise ValueError(
            "training_state.next_epoch exceeds max_epochs"
        )
    if next_epoch == epoch_limit:
        if next_batch != 0:
            raise ValueError(
                "training_state.next_batch_index must be zero after "
                "the final epoch"
            )
        expected_global_step = epoch_limit * expected_steps
        expected_coverage = 0.0
    else:
        if next_batch >= expected_steps:
            raise ValueError(
                "training_state.next_batch_index must point inside "
                "the next epoch"
            )
        expected_global_step = (
            next_epoch * expected_steps + next_batch
        )
        expected_coverage = (
            min(next_batch * expected_batch_size, expected_count)
            / expected_count
        )
    if global_step != expected_global_step:
        raise ValueError(
            "training_state.next_epoch/next_batch_index do not point "
            "to global_step"
        )
    if completed_epochs != next_epoch:
        raise ValueError(
            "training_state.completed_epochs does not match next_epoch"
        )
    if coverage != expected_coverage:
        raise ValueError(
            "training_state.current_epoch_coverage does not match "
            "the next batch cursor"
        )
    return {
        "sampler_version": SAMPLER_VERSION,
        "global_step": global_step,
        "next_epoch": next_epoch,
        "next_batch_index": next_batch,
        "sample_count": sample_count,
        "sample_ids_sha256": sample_digest,
        "steps_per_epoch": state_steps,
        "batch_size": state_batch_size,
        "completed_epochs": completed_epochs,
        "current_epoch_coverage": coverage,
    }


def _epoch_training_state(
    *,
    global_step: int,
    epoch: int,
    batch_index: int,
    steps_per_epoch: int,
    batch_size: int,
    sample_ids: tuple[str, ...],
    current_epoch_coverage: float,
) -> dict[str, object]:
    is_epoch_end = batch_index + 1 == steps_per_epoch
    return {
        "sampler_version": SAMPLER_VERSION,
        "global_step": global_step,
        "next_epoch": epoch + 1 if is_epoch_end else epoch,
        "next_batch_index": 0 if is_epoch_end else batch_index + 1,
        "sample_count": len(sample_ids),
        "sample_ids_sha256": _sample_ids_sha256(sample_ids),
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "completed_epochs": epoch + 1 if is_epoch_end else epoch,
        "current_epoch_coverage": (
            0.0 if is_epoch_end else current_epoch_coverage
        ),
    }


def _source_paths(
    cache_root: Path,
    *,
    patch_center_plan: PatchCenterPlan | None = None,
) -> dict[str, Path]:
    package_root = Path(__file__).resolve().parents[1]
    paths = {
        "cache_manifest": cache_root / "manifest.json",
        "model": package_root / "models" / "pgd.py",
        "model_factory": package_root / "models" / "factory.py",
        "conditioning": package_root / "models" / "conditioning.py",
        "blocks": package_root / "models" / "blocks.py",
        "vq": package_root / "models" / "vq.py",
        "losses": package_root / "models" / "losses.py",
        "fps": package_root / "ops" / "fps.py",
        "cuda_fps": package_root / "ops" / "cuda_fps.py",
        "knn": package_root / "ops" / "knn.py",
        "cuda_knn": package_root / "ops" / "cuda_knn.py",
        "indexing": package_root / "ops" / "indexing.py",
        "interpolate": package_root / "ops" / "interpolate.py",
        "chamfer": package_root / "ops" / "chamfer.py",
        "noise": package_root / "data" / "noise.py",
        "pgd_training": package_root / "data" / "pgd_training.py",
        "patch_center_plan": (
            package_root / "data" / "patch_center_plan.py"
        ),
        "training_engine": Path(__file__).resolve().with_name("engine.py"),
        "sampling": Path(__file__).resolve().with_name("sampling.py"),
        "checkpoint": Path(__file__).resolve().with_name("checkpoint.py"),
        "logger": Path(__file__).resolve().with_name("logger.py"),
        "manifest": Path(__file__).resolve().with_name("manifest.py"),
        "runner": Path(__file__).resolve(),
    }
    if patch_center_plan is not None:
        paths.update(
            {
                "patch_center_manifest": (
                    patch_center_plan.root / "manifest.json"
                ),
                "patch_center_indices": (
                    patch_center_plan.root / "centers.npy"
                ),
            }
        )
    return paths


_RESIDUAL_REFINEMENT_PREFIX = "residual_refinement_"


def _configure_trainable_scope(
    model: object,
    trainable_scope: str,
) -> list[jt.Var]:
    if trainable_scope == "all":
        model.train()
        return list(model.parameters())
    if trainable_scope != "residual_refinement":
        raise ValueError(f"unsupported trainable scope: {trainable_scope}")

    model.eval()
    trainable = []
    for name, parameter in model.named_parameters():
        if name.startswith(_RESIDUAL_REFINEMENT_PREFIX):
            parameter.start_grad()
            trainable.append(parameter)
        else:
            parameter.stop_grad()
    if not trainable:
        raise ValueError("model does not contain residual refinement parameters")
    for module_name in (
        "residual_refinement_linear1",
        "residual_refinement_linear2",
    ):
        module = getattr(model, module_name, None)
        if module is None:
            raise ValueError(f"model is missing {module_name}")
        module.train()
    return trainable


def _validate_refinement_checkpoint_keys(
    checkpoint: Path | str,
    model: object,
) -> None:
    payload = jt.load(str(Path(checkpoint)))
    source_state = payload.get("model") if isinstance(payload, Mapping) else None
    if not isinstance(source_state, Mapping):
        raise ValueError("initial checkpoint model state is invalid")
    destination_state = model.state_dict()
    source_keys = set(source_state)
    destination_keys = set(destination_state)
    refinement_keys = {
        name
        for name in destination_keys
        if name.startswith(_RESIDUAL_REFINEMENT_PREFIX)
    }
    if not source_keys <= destination_keys:
        raise ValueError(
            "initial checkpoint contains model keys absent from destination"
        )
    if destination_keys - source_keys != refinement_keys:
        raise ValueError(
            "destination model extras must be exactly the residual refinement "
            "head"
        )


def _run_epoch_training(
    canonical: Mapping[str, object],
    *,
    config_sha256: str,
    cache_manifest: Mapping[str, object],
    samples: list[object],
    run_dir: Path | str,
    resume_checkpoint: Path | str | None,
    initial_checkpoint: Path | str | None,
    patch_center_plan: PatchCenterPlan | None,
    patch_center_source_verification: Mapping[str, object] | None,
) -> dict[str, object]:
    model_config = canonical["model"]
    data_config = canonical["data"]
    training_config = canonical["training"]
    cache_root = Path(data_config["train_cache"])
    sample_ids = tuple(sample.sample_id for sample in samples)
    sample_count = len(sample_ids)
    batch_size = int(training_config["batch_size"])
    steps_per_epoch = math.ceil(sample_count / batch_size)
    max_epochs = int(training_config["max_epochs"])
    max_steps = steps_per_epoch * max_epochs

    use_cuda = bool(training_config["use_cuda"])
    if use_cuda and not jt.has_cuda:
        raise RuntimeError("training config requires CUDA but Jittor has no CUDA")
    previous_cuda = int(jt.flags.use_cuda)
    jt.flags.use_cuda = int(use_cuda)
    jt.set_global_seed(int(training_config["seed"]))
    try:
        model = build_pgd_model(model_config, data_config)
        trainable_scope = str(training_config.get("trainable_scope", "all"))
        trainable_parameters = _configure_trainable_scope(
            model,
            trainable_scope,
        )
        optimizer = jt.optim.Adam(
            trainable_parameters,
            lr=float(training_config["learning_rate"]),
        )
        initial_state: dict[str, object] = {
            "sampler_version": SAMPLER_VERSION,
            "global_step": 0,
            "next_epoch": 0,
            "next_batch_index": 0,
            "sample_count": sample_count,
            "sample_ids_sha256": _sample_ids_sha256(sample_ids),
            "steps_per_epoch": steps_per_epoch,
            "batch_size": batch_size,
            "completed_epochs": 0,
            "current_epoch_coverage": 0.0,
        }
        initialized_from: dict[str, object] | None = None
        if resume_checkpoint is None:
            if initial_checkpoint is not None:
                if trainable_scope == "residual_refinement":
                    _validate_refinement_checkpoint_keys(
                        initial_checkpoint,
                        model,
                    )
                initialized_from = load_checkpoint(
                    initial_checkpoint,
                    model=model,
                )
            training_state = _validated_epoch_training_state(
                initial_state,
                checkpoint_step=0,
                sample_ids=sample_ids,
                steps_per_epoch=steps_per_epoch,
                batch_size=batch_size,
                max_epochs=max_epochs,
            )
        else:
            loaded = load_checkpoint(
                resume_checkpoint,
                model=model,
                optimizer=optimizer,
                expected_config_sha256=config_sha256,
            )
            if loaded.get("training_state") is None:
                raise ValueError(
                    "v2 resume checkpoint must contain training_state"
                )
            training_state = _validated_epoch_training_state(
                loaded["training_state"],
                checkpoint_step=int(loaded["step"]),
                sample_ids=sample_ids,
                steps_per_epoch=steps_per_epoch,
                batch_size=batch_size,
                max_epochs=max_epochs,
            )
        start_step = int(training_state["global_step"])
        start_epoch = int(training_state["next_epoch"])
        start_batch_index = int(training_state["next_batch_index"])
        if start_step >= max_steps:
            raise ValueError(
                "resume checkpoint step must be smaller than max_steps"
            )

        run_path = Path(run_dir)
        manifest = create_run(
            run_path,
            config=canonical,
            seed=int(training_config["seed"]),
            split_sha256=str(cache_manifest["split_sha256"]),
            source_paths=_source_paths(
                cache_root,
                patch_center_plan=patch_center_plan,
            ),
            environment={
                "jittor_version": jt.__version__,
                "use_cuda": use_cuda,
                "config_sha256": config_sha256,
                "cache_content_sha256": cache_manifest[
                    "content_sha256"
                ],
                "resume_checkpoint": (
                    str(Path(resume_checkpoint).resolve())
                    if resume_checkpoint is not None
                    else None
                ),
                "initial_checkpoint": (
                    str(Path(initial_checkpoint).resolve())
                    if initial_checkpoint is not None
                    else None
                ),
                "initial_checkpoint_step": (
                    int(initialized_from["step"])
                    if initialized_from is not None
                    else None
                ),
                "initial_checkpoint_sha256": (
                    str(initialized_from["checkpoint_sha256"])
                    if initialized_from is not None
                    else None
                ),
                "sampler": SAMPLER_VERSION,
                "sample_ids_sha256": _sample_ids_sha256(sample_ids),
                "steps_per_epoch": steps_per_epoch,
                "max_steps": max_steps,
                "patch_center_mode": (
                    PATCH_CENTER_MODE
                    if patch_center_plan is not None
                    else "random"
                ),
                "patch_center_content_sha256": (
                    patch_center_plan.content_sha256
                    if patch_center_plan is not None
                    else None
                ),
                "patch_center_epoch_offset": (
                    int(data_config["patch_center"]["epoch_offset"])
                    if patch_center_plan is not None
                    else None
                ),
                "patch_center_count": (
                    patch_center_plan.center_count
                    if patch_center_plan is not None
                    else None
                ),
                "patch_center_source_files_verified": (
                    patch_center_source_verification is not None
                ),
                "patch_center_source_file_count": (
                    int(
                        patch_center_source_verification[
                            "verified_file_count"
                        ]
                    )
                    if patch_center_source_verification is not None
                    else None
                ),
                "patch_center_source_file_bytes": (
                    int(
                        patch_center_source_verification[
                            "verified_file_bytes"
                        ]
                    )
                    if patch_center_source_verification is not None
                    else None
                ),
                **(
                    {
                        "surface_view_mode": (
                            SURFACE_VIEW_MODE
                            if "surface_view" in data_config
                            else "view0_only"
                        ),
                        "surface_view_epoch_offset": (
                            int(
                                data_config["surface_view"][
                                    "view_epoch_offset"
                                ]
                            )
                            if "surface_view" in data_config
                            else None
                        ),
                        "surface_view_count": samples[0].view_count,
                    }
                    if (
                        "surface_view" in data_config
                        or cache_manifest.get("format")
                        == "pcdenoise_train_surface_cache_v2"
                    )
                    else {}
                ),
            },
        )
        checkpoints_dir = run_path / "checkpoints"
        checkpoints_dir.mkdir()
        started = time.perf_counter()
        last_metrics: dict[str, float] = {}
        last_checkpoint: dict[str, object] | None = None
        final_epoch_coverage: dict[str, int | float] | None = None
        sample_id_to_index = {
            sample_id: index
            for index, sample_id in enumerate(sample_ids)
        }

        with manifest:
            with (
                TensorBoardLogger(run_path) as logger,
                ExitStack() as prefetch_stack,
            ):
                for epoch in range(start_epoch, max_epochs):
                    planned_batches = epoch_batches(
                        sample_count,
                        batch_size=batch_size,
                        seed=int(training_config["seed"]),
                        epoch=epoch,
                    )
                    if len(planned_batches) != steps_per_epoch:
                        raise RuntimeError(
                            "epoch sampler produced an invalid batch count"
                        )
                    first_batch = (
                        start_batch_index if epoch == start_epoch else 0
                    )
                    observed_ids = [
                        sample_ids[int(index)]
                        for planned in planned_batches[:first_batch]
                        for index in planned
                    ]

                    def build_batch(batch_index: int):
                        return build_epoch_training_batch(
                            samples,
                            base_seed=int(training_config["seed"]),
                            epoch=epoch,
                            batch_index=batch_index,
                            batch_size=batch_size,
                            patch_size=int(model_config["patch_size"]),
                            noise_profile=str(
                                data_config["noise_profile"]
                            ),
                            noise_min=float(data_config["noise_min"]),
                            noise_max=float(data_config["noise_max"]),
                            rotate=bool(data_config["rotate"]),
                            rotation_mode=str(
                                data_config.get(
                                    "rotation_mode",
                                    "euler_xyz",
                                )
                            ),
                            normalization_mode=str(
                                data_config.get(
                                    "normalization_mode",
                                    "clean_unit",
                                )
                            ),
                            patch_center_indices=(
                                patch_center_plan.indices
                                if patch_center_plan is not None
                                else None
                            ),
                            patch_center_epoch_offset=(
                                int(
                                    data_config["patch_center"][
                                        "epoch_offset"
                                    ]
                                )
                                if patch_center_plan is not None
                                else 0
                            ),
                            surface_view_epoch_offset=(
                                int(
                                    data_config["surface_view"][
                                        "view_epoch_offset"
                                    ]
                                )
                                if "surface_view" in data_config
                                else None
                            ),
                        )

                    batch_stream = prefetch_stack.enter_context(
                        _ordered_prefetch_batches(
                            range(first_batch, steps_per_epoch),
                            build_batch,
                            prefetch_batches=int(
                                training_config.get(
                                    "prefetch_batches",
                                    0,
                                )
                            ),
                        )
                    )
                    for batch_index, batch in batch_stream:
                        step = epoch * steps_per_epoch + batch_index + 1
                        actual_learning_rate = _learning_rate_at_step(
                            training_config,
                            step=step,
                            max_steps=max_steps,
                            steps_per_epoch=steps_per_epoch,
                        )
                        _set_optimizer_learning_rate(
                            optimizer,
                            actual_learning_rate,
                        )
                        expected_ids = tuple(
                            sample_ids[int(index)]
                            for index in planned_batches[batch_index]
                        )
                        expected_prefix_indices = [
                            int(index)
                            for planned in planned_batches[
                                : batch_index + 1
                            ]
                            for index in planned
                        ]
                        expected_prefix = coverage_statistics(
                            expected_prefix_indices,
                            num_items=sample_count,
                        )
                        if (
                            batch.sample_ids != expected_ids
                            or len(batch.noisy) != len(expected_ids)
                            or len(batch.clean) != len(expected_ids)
                            or batch.epoch != epoch
                            or batch.batch_index != batch_index
                            or batch.epoch_batch_count
                            != steps_per_epoch
                            or batch.epoch_size != sample_count
                            or batch.prefix_count
                            != len(expected_prefix_indices)
                            or batch.prefix_coverage != expected_prefix
                        ):
                            raise RuntimeError(
                                "epoch training batch violates the "
                                "without-replacement plan"
                            )
                        if "surface_view" in data_config:
                            absolute_view_epoch = (
                                int(
                                    data_config["surface_view"][
                                        "view_epoch_offset"
                                    ]
                                )
                                + epoch
                            )
                            expected_view_id = (
                                absolute_view_epoch % samples[0].view_count
                            )
                            expected_view_visit = (
                                absolute_view_epoch // samples[0].view_count
                            )
                        else:
                            expected_view_id = 0
                            expected_view_visit = epoch
                        if (
                            batch.view_ids
                            != (expected_view_id,) * len(expected_ids)
                            or batch.view_visits
                            != (expected_view_visit,) * len(expected_ids)
                        ):
                            raise RuntimeError(
                                "epoch training batch surface views do not "
                                "match the deterministic schedule"
                            )
                        configured_profile = str(
                            data_config["noise_profile"]
                        )
                        if batch.noise_profiles != (
                            configured_profile,
                        ) * len(expected_ids):
                            raise RuntimeError(
                                "epoch training batch noise profiles "
                                "do not match the config"
                            )
                        if (
                            batch.noise_scales is None
                            or batch.noise_scales.shape
                            != (len(expected_ids),)
                        ):
                            raise RuntimeError(
                                "epoch training batch noise scales "
                                "are invalid"
                            )
                        if (
                            batch.source_noise_scales is None
                            or batch.source_noise_scales.shape
                            != (len(expected_ids),)
                            or batch.normalization_scales is None
                            or batch.normalization_scales.shape
                            != (len(expected_ids),)
                            or not np.isfinite(
                                batch.source_noise_scales
                            ).all()
                            or not np.isfinite(
                                batch.normalization_scales
                            ).all()
                            or float(batch.source_noise_scales.min())
                            <= 0.0
                            or float(batch.normalization_scales.min())
                            <= 0.0
                        ):
                            raise RuntimeError(
                                "epoch training batch alignment metadata "
                                "are invalid"
                            )
                        if bool(
                            model_config.get(
                                "noise_conditioning",
                                False,
                            )
                        ):
                            _validate_effective_noise_scales(
                                batch.noise_scales,
                                data_config=data_config,
                                epoch=epoch,
                                batch_index=batch_index,
                                global_step=step,
                                sample_ids=batch.sample_ids,
                                source_noise_scales=(
                                    batch.source_noise_scales
                                ),
                                normalization_scales=(
                                    batch.normalization_scales
                                ),
                                center_indices=batch.center_indices,
                                view_ids=batch.view_ids,
                            )
                        observed_ids.extend(batch.sample_ids)
                        is_epoch_end = (
                            batch_index + 1 == steps_per_epoch
                        )
                        if is_epoch_end:
                            observed_indices = [
                                sample_id_to_index[sample_id]
                                for sample_id in observed_ids
                            ]
                            coverage = coverage_statistics(
                                observed_indices,
                                num_items=sample_count,
                            )
                            if (
                                coverage.unique_count != sample_count
                                or coverage.duplicate_count != 0
                                or coverage.missing_count != 0
                                or coverage.coverage_fraction != 1.0
                            ):
                                raise RuntimeError(
                                    "epoch did not reach exact 100% coverage"
                                )
                            final_epoch_coverage = coverage.as_dict()

                        noisy = jt.array(batch.noisy)
                        clean = jt.array(batch.clean)
                        prediction, details, teacher_noise_scale = (
                            _epoch_model_forward(
                                model,
                                noisy,
                                noise_scales=batch.noise_scales,
                                model_config=model_config,
                            )
                        )
                        scale_terms = _conditioned_scale_terms(
                            details,
                            teacher_noise_scale,
                        )
                        gate = _displacement_gate(
                            details,
                            enabled=bool(
                                model_config.get(
                                    "pointwise_displacement_gate",
                                    False,
                                )
                            ),
                        )
                        legacy_loss_enabled = (
                            float(
                                training_config[
                                    "reconstruction_weight"
                                ]
                            )
                            > 0.0
                            or float(
                                training_config["point_mse_weight"]
                            )
                            > 0.0
                        )
                        if legacy_loss_enabled:
                            profile = str(
                                training_config["loss_profile"]
                            )
                            reconstruction = infocd_loss(
                                prediction,
                                clean,
                                profile=profile,
                                alpha=(
                                    float(
                                        training_config["paper_alpha"]
                                    )
                                    if profile == "paper"
                                    else None
                                ),
                                validate_finite=False,
                            )
                            point_mse = (
                                (prediction - clean) ** 2
                            ).mean()
                        else:
                            reconstruction = jt.array(0.0).float32()
                            point_mse = jt.array(0.0).float32()
                        commitment = details["commitment_loss"]
                        chamfer_terms = _optional_chamfer_terms(
                            prediction,
                            clean,
                            training_config,
                        )
                        relative_squared_chamfer = (
                            _optional_relative_squared_chamfer(
                                prediction,
                                noisy,
                                clean,
                                training_config,
                            )
                        )
                        huber_loss = None
                        weighted_huber_loss = None
                        if (
                            float(
                                training_config.get(
                                    "correspondence_huber_weight",
                                    0.0,
                                )
                            )
                            > 0.0
                        ):
                            huber_loss = correspondence_huber_loss(
                                prediction,
                                clean,
                                delta=float(
                                    training_config[
                                        "correspondence_huber_delta"
                                    ]
                                ),
                                validate_finite=False,
                            )
                        total_loss = (
                            reconstruction
                            * float(
                                training_config[
                                    "reconstruction_weight"
                                ]
                            )
                            + point_mse
                            * float(
                                training_config["point_mse_weight"]
                            )
                            + commitment
                        )
                        if chamfer_terms is not None:
                            squared_chamfer, target_coverage = (
                                chamfer_terms
                            )
                            total_loss = (
                                total_loss
                                + squared_chamfer
                                * float(
                                    training_config.get(
                                        "squared_chamfer_weight",
                                        0.0,
                                    )
                                )
                                + target_coverage
                                * float(
                                    training_config.get(
                                        "target_coverage_weight",
                                        0.0,
                                    )
                                )
                            )
                        if relative_squared_chamfer is not None:
                            total_loss = (
                                total_loss
                                + relative_squared_chamfer
                                * float(
                                    training_config[
                                        "relative_squared_chamfer_weight"
                                    ]
                                )
                            )
                        if scale_terms is not None:
                            scale_loss, _estimated_scale, _scale_error = (
                                scale_terms
                            )
                            total_loss = (
                                total_loss
                                + scale_loss
                                * float(
                                    training_config[
                                        "noise_scale_estimation_weight"
                                    ]
                                )
                            )
                        if huber_loss is not None:
                            weighted_huber_loss = (
                                huber_loss
                                * float(
                                    training_config[
                                        "correspondence_huber_weight"
                                    ]
                                )
                            )
                            total_loss = (
                                total_loss
                                + weighted_huber_loss
                            )
                        optimizer.step(total_loss)
                        ema_updates = (
                            0
                            if trainable_scope == "residual_refinement"
                            else model.apply_pending_ema()
                        )
                        profile_fraction = (
                            sum(
                                item == configured_profile
                                for item in batch.noise_profiles
                            )
                            / len(batch.noise_profiles)
                        )
                        epoch_progress = (
                            int(batch.prefix_count) / sample_count
                        )
                        epoch_coverage = float(
                            batch.prefix_coverage.coverage_fraction
                        )
                        should_log = (
                            step == start_step + 1
                            or step
                            % int(training_config["log_every"])
                            == 0
                            or step == max_steps
                            or is_epoch_end
                        )
                        should_checkpoint = (
                            step
                            % int(
                                training_config["checkpoint_every"]
                            )
                            == 0
                            or step == max_steps
                            or is_epoch_end
                        )
                        current_metrics: dict[str, float] | None = None
                        if should_log or should_checkpoint:
                            mean_noise_scale = float(
                                batch.noise_scales.mean()
                            )
                            mean_source_noise_scale = float(
                                batch.source_noise_scales.mean()
                            )
                            current_metrics = {
                                "total_loss": scalar_to_float(
                                    total_loss
                                ),
                                "reconstruction_loss": scalar_to_float(
                                    reconstruction
                                ),
                                "point_mse": scalar_to_float(
                                    point_mse
                                ),
                                "commitment_loss": scalar_to_float(
                                    commitment
                                ),
                                "mean_noise_scale": mean_noise_scale,
                                "min_noise_scale": float(
                                    batch.noise_scales.min()
                                ),
                                "max_noise_scale": float(
                                    batch.noise_scales.max()
                                ),
                                "mean_source_noise_scale": (
                                    mean_source_noise_scale
                                ),
                                "mean_normalization_scale": float(
                                    batch.normalization_scales.mean()
                                ),
                                "learning_rate": actual_learning_rate,
                                "ema_updates": float(ema_updates),
                                "epoch_index": float(epoch),
                                "epoch_progress": epoch_progress,
                                "epoch_coverage": epoch_coverage,
                                "noise_profile_fraction": (
                                    profile_fraction
                                ),
                                "patch_center_epoch_rank": (
                                    float(
                                        int(
                                            data_config["patch_center"][
                                                "epoch_offset"
                                            ]
                                        )
                                        + expected_view_visit
                                    )
                                    if patch_center_plan is not None
                                    else -1.0
                                ),
                                "patch_center_plan_enabled": float(
                                    patch_center_plan is not None
                                ),
                            }
                            if (
                                "surface_view" in data_config
                                or cache_manifest.get("format")
                                == "pcdenoise_train_surface_cache_v2"
                            ):
                                current_metrics.update(
                                    {
                                        "surface_view_id": float(
                                            expected_view_id
                                        ),
                                        "surface_view_visit": float(
                                            expected_view_visit
                                        ),
                                    }
                                )
                            if configured_profile == "gaussian":
                                current_metrics["mean_sigma"] = (
                                    mean_noise_scale
                                )
                            elif configured_profile == "starter_laplace":
                                current_metrics["mean_laplace_b"] = (
                                    mean_source_noise_scale
                                )
                            if chamfer_terms is not None:
                                current_metrics.update(
                                    {
                                        "squared_chamfer": (
                                            scalar_to_float(
                                                squared_chamfer
                                            )
                                        ),
                                        "target_coverage": (
                                            scalar_to_float(
                                                target_coverage
                                            )
                                        ),
                                    }
                                )
                            if relative_squared_chamfer is not None:
                                current_metrics[
                                    "relative_squared_chamfer"
                                ] = scalar_to_float(
                                    relative_squared_chamfer
                                )
                            if scale_terms is not None:
                                scale_loss, estimated_scale, scale_error = (
                                    scale_terms
                                )
                                scale_weight = float(
                                    training_config[
                                        "noise_scale_estimation_weight"
                                    ]
                                )
                                current_metrics.update(
                                    {
                                        "noise_scale_estimation_loss": (
                                            scalar_to_float(scale_loss)
                                        ),
                                        "weighted_noise_scale_estimation_loss": (
                                            scalar_to_float(
                                                scale_loss * scale_weight
                                            )
                                        ),
                                        "mean_estimated_noise_scale": (
                                            scalar_to_float(
                                                estimated_scale.mean()
                                            )
                                        ),
                                        "min_estimated_noise_scale": (
                                            scalar_to_float(
                                                estimated_scale.min()
                                            )
                                        ),
                                        "max_estimated_noise_scale": (
                                            scalar_to_float(
                                                estimated_scale.max()
                                            )
                                        ),
                                        "mean_noise_scale_error": (
                                            scalar_to_float(
                                                scale_error.mean()
                                            )
                                        ),
                                        "min_noise_scale_error": (
                                            scalar_to_float(
                                                scale_error.min()
                                            )
                                        ),
                                        "max_noise_scale_error": (
                                            scalar_to_float(
                                                scale_error.max()
                                            )
                                        ),
                                    }
                                )
                            if huber_loss is not None:
                                current_metrics[
                                    "correspondence_huber_loss"
                                ] = scalar_to_float(huber_loss)
                                current_metrics[
                                    "weighted_correspondence_huber_loss"
                                ] = scalar_to_float(weighted_huber_loss)
                            if gate is not None:
                                current_metrics.update(
                                    {
                                        "mean_displacement_gate": (
                                            scalar_to_float(gate.mean())
                                        ),
                                        "min_displacement_gate": (
                                            scalar_to_float(gate.min())
                                        ),
                                        "max_displacement_gate": (
                                            scalar_to_float(gate.max())
                                        ),
                                    }
                                )
                            last_metrics = current_metrics
                        if should_log:
                            if current_metrics is None:
                                raise RuntimeError(
                                    "logging metrics were not materialized"
                                )
                            for name, value in current_metrics.items():
                                logger.log_scalar(
                                    f"train/{name}",
                                    value,
                                    step=step,
                                )
                            logger.log_scalar(
                                "train/loss",
                                current_metrics["total_loss"],
                                step=step,
                            )
                            logger.log_scalar(
                                "train/noise_profile_fraction/"
                                f"{configured_profile}",
                                profile_fraction,
                                step=step,
                            )
                            logger.log_histogram(
                                "train/noise_scale",
                                batch.noise_scales,
                                step=step,
                            )
                            logger.log_histogram(
                                "train/source_noise_scale",
                                batch.source_noise_scales,
                                step=step,
                            )
                            logger.log_histogram(
                                "train/normalization_scale",
                                batch.normalization_scales,
                                step=step,
                            )
                            if scale_terms is not None:
                                _scale_loss, estimated_scale, scale_error = (
                                    scale_terms
                                )
                                logger.log_histogram(
                                    "train/estimated_noise_scale",
                                    estimated_scale,
                                    step=step,
                                )
                                logger.log_histogram(
                                    "train/noise_scale_error",
                                    scale_error,
                                    step=step,
                                )
                            if gate is not None:
                                logger.log_histogram(
                                    "train/displacement_gate",
                                    gate,
                                    step=step,
                                )
                            for (
                                name,
                                value,
                            ) in model.codebook_metrics().items():
                                logger.log_scalar(
                                    f"vq/{name}",
                                    value,
                                    step=step,
                                )

                        should_visualize = (
                            step == start_step + 1
                            or step
                            % int(
                                training_config["visualize_every"]
                            )
                            == 0
                            or step == max_steps
                        )
                        if should_visualize:
                            logger.log_point_cloud(
                                "points/noisy",
                                batch.noisy[0],
                                step=step,
                            )
                            logger.log_point_cloud(
                                "points/target",
                                batch.clean[0],
                                step=step,
                            )
                            logger.log_point_cloud(
                                "points/prediction",
                                prediction[0],
                                step=step,
                            )

                        state = _epoch_training_state(
                            global_step=step,
                            epoch=epoch,
                            batch_index=batch_index,
                            steps_per_epoch=steps_per_epoch,
                            batch_size=batch_size,
                            sample_ids=sample_ids,
                            current_epoch_coverage=epoch_coverage,
                        )
                        state = _validated_epoch_training_state(
                            state,
                            checkpoint_step=step,
                            sample_ids=sample_ids,
                            steps_per_epoch=steps_per_epoch,
                            batch_size=batch_size,
                            max_epochs=max_epochs,
                        )
                        if should_checkpoint:
                            if current_metrics is None:
                                raise RuntimeError(
                                    "checkpoint metrics were not "
                                    "materialized"
                                )
                            last_checkpoint = save_checkpoint(
                                checkpoints_dir
                                / f"step_{step:08d}.pkl",
                                model=model,
                                optimizer=optimizer,
                                step=step,
                                config_sha256=config_sha256,
                                metrics=current_metrics,
                                training_state=state,
                            )
                        if (
                            should_log
                            or should_visualize
                            or should_checkpoint
                        ):
                            logger.flush()
                    prefetch_stack.close()

                elapsed = time.perf_counter() - started
                if last_checkpoint is None:
                    raise RuntimeError("final checkpoint was not written")
                if final_epoch_coverage is None:
                    raise RuntimeError(
                        "final epoch coverage was not materialized"
                    )
                result: dict[str, object] = {
                    "last_step": max_steps,
                    "max_steps": max_steps,
                    "start_step": start_step,
                    "last_epoch": max_epochs,
                    "completed_epochs": max_epochs,
                    "start_epoch": start_epoch,
                    "start_batch_index": start_batch_index,
                    "steps_per_epoch": steps_per_epoch,
                    "final_epoch_coverage": final_epoch_coverage,
                    "elapsed_seconds": elapsed,
                    "steps_per_second": (
                        (max_steps - start_step) / elapsed
                    ),
                    "last_metrics": last_metrics,
                    "checkpoint": last_checkpoint,
                    "config_sha256": config_sha256,
                    "cache_content_sha256": cache_manifest[
                        "content_sha256"
                    ],
                    "sample_count": sample_count,
                    "sample_ids_sha256": _sample_ids_sha256(sample_ids),
                    "sampler": SAMPLER_VERSION,
                    "patch_center_mode": (
                        PATCH_CENTER_MODE
                        if patch_center_plan is not None
                        else "random"
                    ),
                    "patch_center_content_sha256": (
                        patch_center_plan.content_sha256
                        if patch_center_plan is not None
                        else None
                    ),
                    "patch_center_source_files_verified": (
                        patch_center_source_verification is not None
                    ),
                    **(
                        {
                            "surface_view_mode": (
                                SURFACE_VIEW_MODE
                                if "surface_view" in data_config
                                else "view0_only"
                            ),
                            "surface_view_epoch_offset": (
                                int(
                                    data_config["surface_view"][
                                        "view_epoch_offset"
                                    ]
                                )
                                if "surface_view" in data_config
                                else None
                            ),
                            "surface_view_count": samples[0].view_count,
                        }
                        if (
                            "surface_view" in data_config
                            or cache_manifest.get("format")
                            == "pcdenoise_train_surface_cache_v2"
                        )
                        else {}
                    ),
                }
                manifest.complete(result)
                return result
    finally:
        jt.sync_all()
        jt.flags.use_cuda = previous_cuda


def run_training(
    config: Mapping[str, object],
    *,
    run_dir: Path | str,
    resume_checkpoint: Path | str | None = None,
    initial_checkpoint: Path | str | None = None,
) -> dict[str, object]:
    """Run a bounded PGD experiment and return its terminal result."""

    if resume_checkpoint is not None and initial_checkpoint is not None:
        raise ValueError(
            "resume_checkpoint and initial_checkpoint are mutually exclusive"
        )

    canonical = validate_training_config(config)
    config_sha256 = canonical_config_sha256(canonical)
    model_config = canonical["model"]
    data_config = canonical["data"]
    training_config = canonical["training"]
    if (
        training_config.get("trainable_scope") == "residual_refinement"
        and resume_checkpoint is None
        and initial_checkpoint is None
    ):
        raise ValueError(
            "training.trainable_scope='residual_refinement' requires an "
            "initial_checkpoint"
        )
    cache_root = Path(data_config["train_cache"])
    samples = scan_clean_cache(cache_root)
    cache_manifest = _cache_manifest(cache_root, samples)
    if (
        cache_manifest.get("format")
        == "pcdenoise_train_surface_cache_v2"
        and not bool(data_config["verify_cache"])
    ):
        raise ValueError(
            "surface-cache v2 training requires verify_cache=true and "
            "full expected-content/file verification"
        )
    _verify_formal_cache(canonical)
    if canonical["format"] == EPOCH_TRAIN_CONFIG_FORMAT:
        patch_center_plan: PatchCenterPlan | None = None
        patch_center_source_verification: dict[str, object] | None = None
        if "patch_center" in data_config:
            patch_center_config = data_config["patch_center"]
            patch_center_plan = verify_patch_center_plan(
                patch_center_config["plan_dir"],
                expected_content_sha256=patch_center_config[
                    "expected_content_sha256"
                ],
                source_cache_content_sha256=cache_manifest[
                    "content_sha256"
                ],
                expected_sample_ids=tuple(
                    sample.sample_id for sample in samples
                ),
                expected_point_count=cache_manifest.get("num_points"),
                expected_view_count=samples[0].view_count,
            )
            if "surface_view" in data_config:
                view_epoch_offset = int(
                    data_config["surface_view"]["view_epoch_offset"]
                )
                maximum_visit = (
                    view_epoch_offset
                    + int(training_config["max_epochs"])
                    - 1
                ) // samples[0].view_count
            else:
                maximum_visit = int(training_config["max_epochs"]) - 1
            required_centers = (
                int(patch_center_config["epoch_offset"])
                + maximum_visit
                + 1
            )
            if required_centers > patch_center_plan.center_count:
                raise ValueError(
                    "patch center plan center_count is smaller than "
                    "the required non-cycling view visits: "
                    f"{patch_center_plan.center_count} < {required_centers}"
                )
            patch_center_source_verification = (
                verify_patch_center_source_cache(
                    cache_root,
                    expected_content_sha256=(
                        patch_center_plan.source_cache_content_sha256
                    ),
                    expected_source_views_sha256=(
                        patch_center_plan.source_views_sha256
                    ),
                )
            )
        return _run_epoch_training(
            canonical,
            config_sha256=config_sha256,
            cache_manifest=cache_manifest,
            samples=samples,
            run_dir=run_dir,
            resume_checkpoint=resume_checkpoint,
            initial_checkpoint=initial_checkpoint,
            patch_center_plan=patch_center_plan,
            patch_center_source_verification=(
                patch_center_source_verification
            ),
        )

    use_cuda = bool(training_config["use_cuda"])
    if use_cuda and not jt.has_cuda:
        raise RuntimeError("training config requires CUDA but Jittor has no CUDA")
    previous_cuda = int(jt.flags.use_cuda)
    jt.flags.use_cuda = int(use_cuda)
    jt.set_global_seed(int(training_config["seed"]))
    try:
        model = build_pgd_model(model_config, data_config)
        model.train()
        optimizer = jt.optim.Adam(
            model.parameters(),
            lr=float(training_config["learning_rate"]),
        )
        start_step = 0
        initialized_from: dict[str, object] | None = None
        if resume_checkpoint is not None:
            loaded = load_checkpoint(
                resume_checkpoint,
                model=model,
                optimizer=optimizer,
                expected_config_sha256=config_sha256,
            )
            start_step = int(loaded["step"])
        elif initial_checkpoint is not None:
            initialized_from = load_checkpoint(
                initial_checkpoint,
                model=model,
            )
        max_steps = int(training_config["max_steps"])
        if start_step >= max_steps:
            raise ValueError(
                "resume checkpoint step must be smaller than max_steps"
            )

        run_path = Path(run_dir)
        manifest = create_run(
            run_path,
            config=canonical,
            seed=int(training_config["seed"]),
            split_sha256=str(cache_manifest["split_sha256"]),
            source_paths=_source_paths(cache_root),
            environment={
                "jittor_version": jt.__version__,
                "use_cuda": use_cuda,
                "config_sha256": config_sha256,
                "cache_content_sha256": cache_manifest[
                    "content_sha256"
                ],
                "resume_checkpoint": (
                    str(Path(resume_checkpoint).resolve())
                    if resume_checkpoint is not None
                    else None
                ),
                "initial_checkpoint": (
                    str(Path(initial_checkpoint).resolve())
                    if initial_checkpoint is not None
                    else None
                ),
                "initial_checkpoint_step": (
                    int(initialized_from["step"])
                    if initialized_from is not None
                    else None
                ),
                "initial_checkpoint_sha256": (
                    str(initialized_from["checkpoint_sha256"])
                    if initialized_from is not None
                    else None
                ),
            },
        )
        checkpoints_dir = run_path / "checkpoints"
        checkpoints_dir.mkdir()
        started = time.perf_counter()
        last_metrics: dict[str, float] = {}
        last_checkpoint: dict[str, object] | None = None

        with manifest:
            with TensorBoardLogger(run_path) as logger:
                for step in range(start_step + 1, max_steps + 1):
                    batch = build_training_batch(
                        samples,
                        base_seed=int(training_config["seed"]),
                        step=step,
                        batch_size=int(training_config["batch_size"]),
                        patch_size=int(model_config["patch_size"]),
                        noise_min=float(data_config["noise_min"]),
                        noise_max=float(data_config["noise_max"]),
                        rotate=bool(data_config["rotate"]),
                    )
                    noisy = jt.array(batch.noisy)
                    clean = jt.array(batch.clean)
                    prediction, details = model(
                        noisy,
                        return_details=True,
                    )
                    legacy_loss_enabled = (
                        float(
                            training_config["reconstruction_weight"]
                        )
                        > 0.0
                        or float(training_config["point_mse_weight"]) > 0.0
                    )
                    if legacy_loss_enabled:
                        profile = str(training_config["loss_profile"])
                        reconstruction = infocd_loss(
                            prediction,
                            clean,
                            profile=profile,
                            alpha=(
                                float(training_config["paper_alpha"])
                                if profile == "paper"
                                else None
                            ),
                            validate_finite=False,
                        )
                        point_mse = ((prediction - clean) ** 2).mean()
                    else:
                        reconstruction = jt.array(0.0).float32()
                        point_mse = jt.array(0.0).float32()
                    commitment = details["commitment_loss"]
                    chamfer_terms = _optional_chamfer_terms(
                        prediction,
                        clean,
                        training_config,
                    )
                    relative_squared_chamfer = (
                        _optional_relative_squared_chamfer(
                            prediction,
                            noisy,
                            clean,
                            training_config,
                        )
                    )
                    total_loss = (
                        reconstruction
                        * float(training_config["reconstruction_weight"])
                        + point_mse
                        * float(training_config["point_mse_weight"])
                        + commitment
                    )
                    if chamfer_terms is not None:
                        squared_chamfer, target_coverage = chamfer_terms
                        total_loss = (
                            total_loss
                            + squared_chamfer
                            * float(
                                training_config.get(
                                    "squared_chamfer_weight", 0.0
                                )
                            )
                            + target_coverage
                            * float(
                                training_config.get(
                                    "target_coverage_weight", 0.0
                                )
                            )
                        )
                    if relative_squared_chamfer is not None:
                        total_loss = (
                            total_loss
                            + relative_squared_chamfer
                            * float(
                                training_config[
                                    "relative_squared_chamfer_weight"
                                ]
                            )
                        )
                    optimizer.step(total_loss)
                    ema_updates = model.apply_pending_ema()

                    should_log = (
                        step == start_step + 1
                        or step % int(training_config["log_every"]) == 0
                        or step == max_steps
                    )
                    should_checkpoint = (
                        step
                        % int(training_config["checkpoint_every"])
                        == 0
                        or step == max_steps
                    )
                    current_metrics: dict[str, float] | None = None
                    if should_log or should_checkpoint:
                        current_metrics = {
                            "total_loss": scalar_to_float(total_loss),
                            "reconstruction_loss": scalar_to_float(
                                reconstruction
                            ),
                            "point_mse": scalar_to_float(point_mse),
                            "commitment_loss": scalar_to_float(commitment),
                            "mean_sigma": float(batch.sigmas.mean()),
                            "learning_rate": float(
                                training_config["learning_rate"]
                            ),
                            "ema_updates": float(ema_updates),
                        }
                        if chamfer_terms is not None:
                            current_metrics.update(
                                {
                                    "squared_chamfer": scalar_to_float(
                                        squared_chamfer
                                    ),
                                    "target_coverage": scalar_to_float(
                                        target_coverage
                                    ),
                                }
                            )
                        if relative_squared_chamfer is not None:
                            current_metrics[
                                "relative_squared_chamfer"
                            ] = scalar_to_float(
                                relative_squared_chamfer
                            )
                        last_metrics = current_metrics
                    if should_log:
                        if current_metrics is None:
                            raise RuntimeError(
                                "logging metrics were not materialized"
                            )
                        for name, value in current_metrics.items():
                            logger.log_scalar(
                                f"train/{name}",
                                value,
                                step=step,
                            )
                        logger.log_scalar(
                            "train/loss",
                            current_metrics["total_loss"],
                            step=step,
                        )
                        for name, value in model.codebook_metrics().items():
                            logger.log_scalar(
                                f"vq/{name}",
                                value,
                                step=step,
                            )

                    should_visualize = (
                        step == start_step + 1
                        or step
                        % int(training_config["visualize_every"])
                        == 0
                        or step == max_steps
                    )
                    if should_visualize:
                        logger.log_point_cloud(
                            "points/noisy",
                            batch.noisy[0],
                            step=step,
                        )
                        logger.log_point_cloud(
                            "points/target",
                            batch.clean[0],
                            step=step,
                        )
                        logger.log_point_cloud(
                            "points/prediction",
                            prediction[0],
                            step=step,
                        )

                    if should_checkpoint:
                        if current_metrics is None:
                            raise RuntimeError(
                                "checkpoint metrics were not materialized"
                            )
                        last_checkpoint = save_checkpoint(
                            checkpoints_dir
                            / f"step_{step:08d}.pkl",
                            model=model,
                            optimizer=optimizer,
                            step=step,
                            config_sha256=config_sha256,
                            metrics=current_metrics,
                        )
                    if should_log or should_visualize or should_checkpoint:
                        logger.flush()

                elapsed = time.perf_counter() - started
                if last_checkpoint is None:
                    raise RuntimeError("final checkpoint was not written")
                result: dict[str, object] = {
                    "last_step": max_steps,
                    "start_step": start_step,
                    "elapsed_seconds": elapsed,
                    "steps_per_second": (
                        (max_steps - start_step) / elapsed
                    ),
                    "last_metrics": last_metrics,
                    "checkpoint": last_checkpoint,
                    "config_sha256": config_sha256,
                    "cache_content_sha256": cache_manifest[
                        "content_sha256"
                    ],
                    "sample_count": len(samples),
                }
                manifest.complete(result)
                return result
    finally:
        jt.sync_all()
        jt.flags.use_cuda = previous_cuda


__all__ = [
    "EPOCH_TRAIN_CONFIG_FORMAT",
    "TRAIN_CONFIG_FORMAT",
    "run_training",
    "validate_training_config",
]

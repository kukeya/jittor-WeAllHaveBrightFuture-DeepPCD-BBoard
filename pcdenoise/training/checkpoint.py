"""Integrity-checked atomic checkpoints for Jittor training runs."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from numbers import Integral
from pathlib import Path
from typing import Mapping

import jittor as jt


CHECKPOINT_FORMAT = "pcdenoise_jittor_checkpoint_v1"


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _step(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < 0
    ):
        raise ValueError("step must be a nonnegative integer")
    return int(value)


def _json_snapshot(
    value: Mapping[str, object] | None,
    *,
    name: str,
) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{name} must contain finite JSON-compatible values"
        ) from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError(f"{name} must be a mapping")
    return decoded


def _snapshot_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"checkpoint component is not regular: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_unlink_owned(path: Path, *, device: int, inode: int) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    if (
        metadata.st_dev != device
        or metadata.st_ino != inode
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
    ):
        raise RuntimeError(f"refusing to unlink replaced artifact: {path}")
    path.unlink()


def _parameter_stop_grad_mask(model: object) -> dict[str, bool]:
    """Snapshot every named Var's trainable/buffer status.

    Jittor 1.3.10 propagates ``stop_grad`` from the source Var when
    ``Module.load_parameters`` calls ``destination.update(source)``.  Vars
    deserialized by ``jt.load`` are stopped, so loading an otherwise valid
    checkpoint can silently turn every trainable parameter into a buffer.
    Capture the destination model's intended mask before loading and restore
    it afterwards.
    """

    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("model must provide named_parameters()")
    result: dict[str, bool] = {}
    for item in named_parameters():
        if (
            not isinstance(item, (tuple, list))
            or len(item) != 2
            or not isinstance(item[0], str)
        ):
            raise TypeError("model named_parameters() entries are invalid")
        name, parameter = item
        if name in result:
            raise ValueError(f"duplicate named parameter: {name}")
        is_stop_grad = getattr(parameter, "is_stop_grad", None)
        if not callable(is_stop_grad):
            raise TypeError(f"named parameter is not a Jittor Var: {name}")
        result[name] = bool(is_stop_grad())
    if not result:
        raise ValueError("model does not contain named parameters")
    return result


def _restore_parameter_stop_grad_mask(
    model: object,
    expected: Mapping[str, bool],
) -> None:
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("model must provide named_parameters()")
    current = list(named_parameters())
    current_names = [item[0] for item in current]
    if len(current_names) != len(set(current_names)):
        raise ValueError("model contains duplicate named parameters")
    if set(current_names) != set(expected):
        raise ValueError(
            "model parameter names changed while loading checkpoint"
        )
    for name, parameter in current:
        should_stop = expected[name]
        is_stopped = bool(parameter.is_stop_grad())
        if should_stop and not is_stopped:
            parameter.stop_grad()
        elif not should_stop and is_stopped:
            parameter.start_grad()
        if bool(parameter.is_stop_grad()) != should_stop:
            raise RuntimeError(
                f"could not restore gradient status for parameter: {name}"
            )


def save_checkpoint(
    path: os.PathLike[str] | str,
    *,
    model: object,
    optimizer: object | None,
    step: int,
    config_sha256: str,
    metrics: Mapping[str, object] | None = None,
    training_state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Save one no-overwrite checkpoint plus a SHA256 sidecar."""

    checkpoint_path = Path(path)
    parent = checkpoint_path.parent
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    sidecar_path = checkpoint_path.with_name(
        checkpoint_path.name + ".sha256"
    )
    if os.path.lexists(checkpoint_path):
        raise FileExistsError(checkpoint_path)
    if os.path.lexists(sidecar_path):
        raise FileExistsError(sidecar_path)
    actual_step = _step(step)
    config_digest = _sha256(config_sha256, name="config_sha256")
    metric_snapshot = _json_snapshot(metrics, name="metrics")
    has_training_state = training_state is not None
    training_snapshot = (
        _json_snapshot(training_state, name="training_state")
        if has_training_state
        else None
    )
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("model must provide state_dict()")
    optimizer_state = None
    if optimizer is not None:
        optimizer_state_dict = getattr(optimizer, "state_dict", None)
        if not callable(optimizer_state_dict):
            raise TypeError("optimizer must provide state_dict()")
        optimizer_state = optimizer_state_dict()

    payload = {
        "format": CHECKPOINT_FORMAT,
        "format_version": 2 if has_training_state else 1,
        "step": actual_step,
        "config_sha256": config_digest,
        "metrics": metric_snapshot,
        "model": state_dict(),
        "optimizer": optimizer_state,
    }
    if has_training_state:
        payload["training_state"] = training_snapshot
    checkpoint_descriptor, checkpoint_temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{checkpoint_path.name}.",
        suffix=".tmp",
    )
    os.close(checkpoint_descriptor)
    checkpoint_temporary = Path(checkpoint_temporary_name)
    sidecar_temporary: Path | None = None
    published_checkpoint = False
    published_sidecar = False
    checkpoint_identity: tuple[int, int] | None = None
    sidecar_identity: tuple[int, int] | None = None
    try:
        jt.save(payload, str(checkpoint_temporary))
        with checkpoint_temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        checkpoint_bytes = _snapshot_regular_file(checkpoint_temporary)
        digest = hashlib.sha256(checkpoint_bytes).hexdigest()

        sidecar_descriptor, sidecar_temporary_name = tempfile.mkstemp(
            dir=parent,
            prefix=f".{sidecar_path.name}.",
            suffix=".tmp",
        )
        sidecar_temporary = Path(sidecar_temporary_name)
        with os.fdopen(
            sidecar_descriptor, "w", encoding="ascii"
        ) as stream:
            stream.write(digest + "\n")
            stream.flush()
            os.fsync(stream.fileno())

        checkpoint_metadata = os.lstat(checkpoint_temporary)
        checkpoint_identity = (
            checkpoint_metadata.st_dev,
            checkpoint_metadata.st_ino,
        )
        sidecar_metadata = os.lstat(sidecar_temporary)
        sidecar_identity = (
            sidecar_metadata.st_dev,
            sidecar_metadata.st_ino,
        )
        os.link(checkpoint_temporary, checkpoint_path)
        published_checkpoint = True
        os.link(sidecar_temporary, sidecar_path)
        published_sidecar = True
        _fsync_directory(parent)
    except BaseException:
        if (
            published_sidecar
            and sidecar_identity is not None
        ):
            _safe_unlink_owned(
                sidecar_path,
                device=sidecar_identity[0],
                inode=sidecar_identity[1],
            )
        if (
            published_checkpoint
            and checkpoint_identity is not None
        ):
            _safe_unlink_owned(
                checkpoint_path,
                device=checkpoint_identity[0],
                inode=checkpoint_identity[1],
            )
        raise
    finally:
        if checkpoint_temporary.exists():
            checkpoint_temporary.unlink()
        if sidecar_temporary is not None and sidecar_temporary.exists():
            sidecar_temporary.unlink()

    result = {
        "format": CHECKPOINT_FORMAT,
        "step": actual_step,
        "config_sha256": config_digest,
        "metrics": metric_snapshot,
        "checkpoint_sha256": digest,
        "path": str(checkpoint_path),
    }
    if has_training_state:
        result["training_state"] = training_snapshot
    return result


def load_checkpoint(
    path: os.PathLike[str] | str,
    *,
    model: object,
    optimizer: object | None = None,
    expected_config_sha256: str | None = None,
) -> dict[str, object]:
    """Verify a checkpoint snapshot, then restore model/optimizer state."""

    checkpoint_path = Path(path)
    sidecar_path = checkpoint_path.with_name(
        checkpoint_path.name + ".sha256"
    )
    checkpoint_bytes = _snapshot_regular_file(checkpoint_path)
    sidecar_bytes = _snapshot_regular_file(sidecar_path)
    try:
        claimed_digest = sidecar_bytes.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ValueError("checkpoint SHA256 sidecar is not ASCII") from error
    _sha256(claimed_digest, name="checkpoint SHA256 sidecar")
    actual_digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    if actual_digest != claimed_digest:
        raise ValueError("checkpoint SHA256 does not match its sidecar")

    descriptor, snapshot_name = tempfile.mkstemp(
        prefix=".pcdenoise-checkpoint-load.",
        suffix=".pkl",
    )
    snapshot_path = Path(snapshot_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(checkpoint_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        payload = jt.load(str(snapshot_path))
    finally:
        if snapshot_path.exists():
            snapshot_path.unlink()
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    format_version = payload.get("format_version")
    if (
        payload.get("format") != CHECKPOINT_FORMAT
        or isinstance(format_version, bool)
        or format_version not in (1, 2)
    ):
        raise ValueError("checkpoint format is invalid")
    if format_version == 1:
        if "training_state" in payload:
            raise ValueError(
                "legacy checkpoint must not contain training_state"
            )
        training_state = None
    else:
        if "training_state" not in payload:
            raise ValueError(
                "version 2 checkpoint must contain training_state"
            )
        if not isinstance(payload["training_state"], Mapping):
            raise ValueError(
                "checkpoint training_state must be a mapping"
            )
        training_state = _json_snapshot(
            payload["training_state"],
            name="training_state",
        )
    actual_step = _step(payload.get("step"))
    config_digest = _sha256(
        payload.get("config_sha256"),
        name="checkpoint config_sha256",
    )
    if expected_config_sha256 is not None:
        expected = _sha256(
            expected_config_sha256,
            name="expected_config_sha256",
        )
        if config_digest != expected:
            raise ValueError(
                "checkpoint config SHA256 does not match expected config"
            )
    metrics = _json_snapshot(payload.get("metrics"), name="metrics")
    model_state = payload.get("model")
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint model state is invalid")
    optimizer_state = payload.get("optimizer")
    if optimizer is not None and optimizer_state is None:
        raise ValueError("checkpoint does not contain optimizer state")

    load_parameters = getattr(model, "load_parameters", None)
    if not callable(load_parameters):
        raise TypeError("model must provide load_parameters()")
    stop_grad_mask = _parameter_stop_grad_mask(model)
    load_parameters(model_state)
    if optimizer is not None:
        load_optimizer = getattr(optimizer, "load_state_dict", None)
        if not callable(load_optimizer):
            raise TypeError("optimizer must provide load_state_dict()")
        load_optimizer(optimizer_state)
    # Both model and optimizer state loading can propagate the stopped status
    # of deserialized Vars into live parameters.  Restore only after both have
    # finished so the model's original trainable/buffer contract wins.
    _restore_parameter_stop_grad_mask(model, stop_grad_mask)
    return {
        "format": CHECKPOINT_FORMAT,
        "step": actual_step,
        "config_sha256": config_digest,
        "metrics": metrics,
        "training_state": training_state,
        "checkpoint_sha256": actual_digest,
        "path": str(checkpoint_path),
    }


__all__ = [
    "CHECKPOINT_FORMAT",
    "load_checkpoint",
    "save_checkpoint",
]

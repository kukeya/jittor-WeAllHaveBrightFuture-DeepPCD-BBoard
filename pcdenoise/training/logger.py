"""TensorBoardX logging that safely detaches Jittor values to the host."""

from __future__ import annotations

import math
import os
import shutil
import stat
from pathlib import Path

import numpy as np
from tensorboardX import SummaryWriter

from pcdenoise.visualization import prepare_point_cloud


def _validated_tag(tag: str) -> str:
    if not isinstance(tag, str) or not tag.strip() or "\x00" in tag:
        raise ValueError("tag must be a non-empty string")
    return tag


def _validated_step(step: int) -> int:
    if (
        isinstance(step, bool)
        or not isinstance(step, (int, np.integer))
        or step < 0
    ):
        raise ValueError("step must be a nonnegative integer")
    return int(step)


def scalar_to_float(value: object) -> float:
    """Detach one Python/NumPy/Jittor scalar and validate it is finite."""

    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("scalar value must contain exactly one element")
        raw_value = value.item()
    elif isinstance(value, np.generic):
        raw_value = value.item()
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        raw_value = value
    else:
        item_method = getattr(value, "item", None)
        if not callable(item_method):
            raise TypeError("scalar value must provide item() or be numeric")
        raw_value = item_method()
    try:
        result = float(raw_value)
    except (TypeError, ValueError) as error:
        raise TypeError("scalar item must convert to float") from error
    if not math.isfinite(result):
        raise ValueError("scalar value must be finite")
    return result


def _histogram_to_numpy(values: object) -> np.ndarray:
    """Detach, flatten, and validate real numeric histogram values."""

    if isinstance(values, np.ndarray):
        array = values
    else:
        numpy_method = getattr(values, "numpy", None)
        if not callable(numpy_method):
            raise TypeError(
                "histogram values must be a NumPy array or provide numpy()"
            )
        array = np.asarray(numpy_method())
    if array.ndim == 0:
        raise ValueError("histogram values must be at least one-dimensional")
    if array.size == 0:
        raise ValueError("histogram values must be non-empty")
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.complexfloating)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise TypeError("histogram values must be real numeric")
    if not np.isfinite(array).all():
        raise ValueError("histogram values must be finite")
    return np.ascontiguousarray(array.reshape(-1))


class TensorBoardLogger:
    """Write one new run's scalar and point-cloud TensorBoard events."""

    def __init__(
        self,
        run_dir: os.PathLike[str] | str,
        *,
        flush_secs: int = 30,
        max_queue: int = 10,
    ) -> None:
        run_path = Path(run_dir)
        if not run_path.exists():
            raise FileNotFoundError(run_path)
        if not run_path.is_dir():
            raise NotADirectoryError(run_path)
        if (
            isinstance(flush_secs, bool)
            or not isinstance(flush_secs, int)
            or flush_secs <= 0
        ):
            raise ValueError("flush_secs must be a positive integer")
        if (
            isinstance(max_queue, bool)
            or not isinstance(max_queue, int)
            or max_queue <= 0
        ):
            raise ValueError("max_queue must be a positive integer")

        self.run_dir = run_path.resolve()
        self.log_dir = self.run_dir / "tensorboard"
        if self.log_dir.exists():
            raise FileExistsError(
                f"refusing to overwrite TensorBoard directory: {self.log_dir}"
            )
        self.log_dir.mkdir()
        log_metadata = os.lstat(self.log_dir)
        try:
            self._writer = SummaryWriter(
                logdir=str(self.log_dir),
                flush_secs=flush_secs,
                max_queue=max_queue,
            )
        except BaseException:
            try:
                current_metadata = os.lstat(self.log_dir)
            except FileNotFoundError:
                current_metadata = None
            if (
                current_metadata is not None
                and current_metadata.st_dev == log_metadata.st_dev
                and current_metadata.st_ino == log_metadata.st_ino
                and stat.S_ISDIR(current_metadata.st_mode)
                and not stat.S_ISLNK(current_metadata.st_mode)
            ):
                shutil.rmtree(self.log_dir)
            raise
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("TensorBoardLogger is closed")

    def log_scalar(self, tag: str, value: object, *, step: int) -> None:
        """Synchronize one scalar to the host and write it."""

        self._require_open()
        self._writer.add_scalar(
            _validated_tag(tag),
            scalar_to_float(value),
            global_step=_validated_step(step),
        )

    def log_histogram(
        self,
        tag: str,
        values: object,
        *,
        step: int,
    ) -> None:
        """Synchronize real finite values to the host and log a histogram."""

        self._require_open()
        validated_tag = _validated_tag(tag)
        validated_step = _validated_step(step)
        self._writer.add_histogram(
            validated_tag,
            _histogram_to_numpy(values),
            global_step=validated_step,
        )

    def log_point_cloud(
        self,
        tag: str,
        points: object,
        *,
        step: int,
        colors: object | None = None,
    ) -> None:
        """Synchronize a cloud to the host and write a mesh-plugin event."""

        self._require_open()
        vertices, prepared_colors = prepare_point_cloud(
            points, colors=colors
        )
        self._writer.add_mesh(
            _validated_tag(tag),
            vertices,
            colors=prepared_colors,
            global_step=_validated_step(step),
        )

    def flush(self) -> None:
        self._require_open()
        self._writer.flush()

    def close(self) -> None:
        if not self._closed:
            self._writer.close()
            self._closed = True

    def __enter__(self) -> "TensorBoardLogger":
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

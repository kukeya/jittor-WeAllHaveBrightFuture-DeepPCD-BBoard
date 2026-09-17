"""Deterministic without-replacement epoch sampling using only NumPy."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from numbers import Integral
from typing import Iterable

import numpy as np


SAMPLER_VERSION = "epoch_without_replacement_v1"
_SEED_DOMAIN = f"pcdenoise:{SAMPLER_VERSION}".encode("ascii")
_UINT64_LIMIT = 2**64


def _positive_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _uint64(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < 0
        or int(value) >= _UINT64_LIMIT
    ):
        raise ValueError(
            f"{name} must be a nonnegative integer smaller than 2**64"
        )
    return int(value)


def _nonnegative_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) < 0
    ):
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _permutation_seed(seed: int, epoch: int) -> int:
    digest = hashlib.sha256(
        _SEED_DOMAIN
        + seed.to_bytes(8, "little", signed=False)
        + epoch.to_bytes(8, "little", signed=False)
    ).digest()
    return int.from_bytes(digest[:16], "little", signed=False)


@dataclass(frozen=True)
class EpochCoverage:
    """Coverage counters suitable for manifests and scalar logging."""

    unique_count: int
    duplicate_count: int
    missing_count: int
    coverage_fraction: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "unique_count": self.unique_count,
            "duplicate_count": self.duplicate_count,
            "missing_count": self.missing_count,
            "coverage_fraction": self.coverage_fraction,
        }


def epoch_permutation(
    num_items: int,
    *,
    seed: int,
    epoch: int,
) -> np.ndarray:
    """Return the resume-stable index permutation for one epoch."""

    count = _positive_integer(num_items, name="num_items")
    base_seed = _uint64(seed, name="seed")
    epoch_index = _uint64(epoch, name="epoch")
    rng = np.random.default_rng(_permutation_seed(base_seed, epoch_index))
    permutation = np.ascontiguousarray(rng.permutation(count), dtype=np.int64)
    permutation.setflags(write=False)
    return permutation


def epoch_batches(
    num_items: int,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    batch_offset: int = 0,
) -> tuple[np.ndarray, ...]:
    """Return the remaining batches in an epoch without padding the tail.

    ``batch_offset`` counts already completed batches. Passing the total batch
    count therefore returns an empty tuple, which is the completed-epoch state.
    """

    count = _positive_integer(num_items, name="num_items")
    size = _positive_integer(batch_size, name="batch_size")
    offset = _nonnegative_integer(batch_offset, name="batch_offset")
    total_batches = math.ceil(count / size)
    if offset > total_batches:
        raise ValueError(
            "batch_offset must not exceed the number of batches in the epoch"
        )

    permutation = epoch_permutation(count, seed=seed, epoch=epoch)
    return tuple(
        permutation[start : min(start + size, count)]
        for start in range(offset * size, count, size)
    )


def coverage_statistics(
    indices: Iterable[int] | np.ndarray,
    *,
    num_items: int,
) -> EpochCoverage:
    """Measure unique, duplicate, and missing visits against an epoch domain."""

    count = _positive_integer(num_items, name="num_items")
    if isinstance(indices, (str, bytes)):
        raise ValueError("indices must be a one-dimensional integer iterable")
    try:
        raw_values = list(indices)
    except TypeError as error:
        raise ValueError(
            "indices must be a one-dimensional integer iterable"
        ) from error
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in raw_values
    ):
        raise ValueError("indices must be a one-dimensional integer iterable")
    try:
        values = np.asarray(
            [int(value) for value in raw_values],
            dtype=np.int64,
        )
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(
            "indices must be a one-dimensional integer iterable"
        ) from error
    if values.size and ((values < 0).any() or (values >= count).any()):
        raise ValueError(f"indices must be in the range [0, {count})")

    unique_count = int(np.unique(values).size)
    return EpochCoverage(
        unique_count=unique_count,
        duplicate_count=int(values.size) - unique_count,
        missing_count=count - unique_count,
        coverage_fraction=unique_count / count,
    )


__all__ = [
    "SAMPLER_VERSION",
    "EpochCoverage",
    "coverage_statistics",
    "epoch_batches",
    "epoch_permutation",
]

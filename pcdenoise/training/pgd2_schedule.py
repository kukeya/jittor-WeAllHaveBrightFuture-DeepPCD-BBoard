"""Strict, stateless learning-rate schedules for PGD2 training.

Epoch numbers in configuration are one-based.  Training cursors are zero-based,
while the absolute optimizer step returned by :func:`absolute_step_from_cursor`
is one-based.  Keeping those conversions here prevents a resumed run from
restarting warmup or shifting a cosine boundary by one update.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real


CONSTANT_SCHEDULE = "constant"
WARMUP_HOLD_COSINE_SCHEDULE = "warmup_hold_cosine_v1"

_CONFIG_KEYS = frozenset(
    ("learning_rate", "max_epochs", "learning_rate_schedule")
)
_STAGED_SCHEDULE_KEYS = frozenset(
    (
        "name",
        "warmup_start_learning_rate",
        "warmup_epochs",
        "cosine_start_epoch",
        "min_learning_rate",
    )
)


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return dict(value)


def _exact_keys(
    value: Mapping[str, object],
    *,
    expected: frozenset[str],
    name: str,
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing keys: {missing}")
        if extra:
            parts.append(f"unexpected keys: {extra}")
        raise ValueError(f"{name} has " + "; ".join(parts))


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _finite_real(
    value: object,
    *,
    name: str,
    strictly_positive: bool,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        qualifier = "positive" if strictly_positive else "nonnegative"
        raise TypeError(f"{name} must be a finite {qualifier} number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if not strictly_positive and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def canonicalize_pgd2_schedule_config(
    config: object,
) -> dict[str, object]:
    """Validate and canonicalize the complete PGD2 schedule contract.

    The accepted top-level keys are exactly ``learning_rate``, ``max_epochs``,
    and ``learning_rate_schedule``.  ``constant`` may be written as a string or
    as ``{"name": "constant"}``; the canonical representation always uses the
    mapping form.  The staged schedule requires every boundary explicitly so a
    config digest fully determines its behavior.
    """

    source = _mapping(config, name="PGD2 schedule config")
    _exact_keys(source, expected=_CONFIG_KEYS, name="PGD2 schedule config")
    base = _finite_real(
        source["learning_rate"],
        name="learning_rate",
        strictly_positive=True,
    )
    max_epochs = _positive_integer(
        source["max_epochs"],
        name="max_epochs",
    )
    schedule_value = source["learning_rate_schedule"]

    if isinstance(schedule_value, str):
        if schedule_value != CONSTANT_SCHEDULE:
            raise ValueError(
                "non-constant learning_rate_schedule must be an explicit "
                "mapping"
            )
        canonical_schedule: dict[str, object] = {
            "name": CONSTANT_SCHEDULE
        }
    else:
        schedule = _mapping(
            schedule_value,
            name="learning_rate_schedule",
        )
        name = schedule.get("name")
        if name == CONSTANT_SCHEDULE:
            _exact_keys(
                schedule,
                expected=frozenset(("name",)),
                name="constant learning_rate_schedule",
            )
            canonical_schedule = {"name": CONSTANT_SCHEDULE}
        elif name == WARMUP_HOLD_COSINE_SCHEDULE:
            _exact_keys(
                schedule,
                expected=_STAGED_SCHEDULE_KEYS,
                name="warmup_hold_cosine_v1 learning_rate_schedule",
            )
            warmup_start = _finite_real(
                schedule["warmup_start_learning_rate"],
                name="warmup_start_learning_rate",
                strictly_positive=True,
            )
            minimum = _finite_real(
                schedule["min_learning_rate"],
                name="min_learning_rate",
                strictly_positive=False,
            )
            warmup_epochs = _positive_integer(
                schedule["warmup_epochs"],
                name="warmup_epochs",
            )
            cosine_start_epoch = _positive_integer(
                schedule["cosine_start_epoch"],
                name="cosine_start_epoch",
            )
            if warmup_start > base:
                raise ValueError(
                    "warmup_start_learning_rate must not exceed "
                    "learning_rate"
                )
            if minimum > base:
                raise ValueError(
                    "min_learning_rate must not exceed learning_rate"
                )
            if cosine_start_epoch <= warmup_epochs:
                raise ValueError(
                    "cosine_start_epoch must be greater than warmup_epochs"
                )
            if cosine_start_epoch > max_epochs:
                raise ValueError(
                    "cosine_start_epoch must not exceed max_epochs"
                )
            canonical_schedule = {
                "name": WARMUP_HOLD_COSINE_SCHEDULE,
                "warmup_start_learning_rate": warmup_start,
                "warmup_epochs": warmup_epochs,
                "cosine_start_epoch": cosine_start_epoch,
                "min_learning_rate": minimum,
            }
        else:
            raise ValueError(
                "learning_rate_schedule.name must be 'constant' or "
                "'warmup_hold_cosine_v1'"
            )

    return {
        "learning_rate": base,
        "max_epochs": max_epochs,
        "learning_rate_schedule": canonical_schedule,
    }


def absolute_step_from_cursor(
    *,
    epoch_index: object,
    batch_index: object,
    steps_per_epoch: object,
) -> int:
    """Convert a zero-based epoch/batch cursor to a one-based update step."""

    epoch = _nonnegative_integer(epoch_index, name="epoch_index")
    batch = _nonnegative_integer(batch_index, name="batch_index")
    epoch_steps = _positive_integer(
        steps_per_epoch,
        name="steps_per_epoch",
    )
    if batch >= epoch_steps:
        raise ValueError("batch_index must be smaller than steps_per_epoch")
    return epoch * epoch_steps + batch + 1


def absolute_step_from_resume_state(
    *,
    global_step: object,
    next_epoch_index: object,
    next_batch_index: object,
    steps_per_epoch: object,
) -> int:
    """Validate a resume cursor and return its next one-based update step.

    ``global_step`` is the number of optimizer updates already completed.  A
    mismatch is rejected instead of silently selecting an incorrect LR phase.
    """

    completed = _nonnegative_integer(global_step, name="global_step")
    next_step = absolute_step_from_cursor(
        epoch_index=next_epoch_index,
        batch_index=next_batch_index,
        steps_per_epoch=steps_per_epoch,
    )
    if next_step != completed + 1:
        raise ValueError(
            "resume cursor does not point immediately after global_step"
        )
    return next_step


def pgd2_learning_rate_at_step(
    config: object,
    *,
    absolute_step: object,
    steps_per_epoch: object,
) -> float:
    """Return the deterministic LR for one one-based optimizer step.

    Warmup includes both endpoints, the hold phase ends immediately before the
    configured one-based ``cosine_start_epoch``, and cosine also includes both
    endpoints.  This makes the first cosine step exactly the base LR and the
    final step of ``max_epochs`` exactly ``min_learning_rate``.
    """

    canonical = canonicalize_pgd2_schedule_config(config)
    step = _positive_integer(absolute_step, name="absolute_step")
    epoch_steps = _positive_integer(
        steps_per_epoch,
        name="steps_per_epoch",
    )
    max_epochs = int(canonical["max_epochs"])
    final_step = max_epochs * epoch_steps
    if step > final_step:
        raise ValueError("absolute_step must not exceed the configured run")

    base = float(canonical["learning_rate"])
    schedule = canonical["learning_rate_schedule"]
    if schedule["name"] == CONSTANT_SCHEDULE:
        return base

    warmup_start = float(schedule["warmup_start_learning_rate"])
    warmup_final_step = int(schedule["warmup_epochs"]) * epoch_steps
    cosine_first_step = (
        (int(schedule["cosine_start_epoch"]) - 1) * epoch_steps + 1
    )
    minimum = float(schedule["min_learning_rate"])

    if warmup_final_step == 1 and warmup_start != base:
        raise ValueError(
            "warmup needs at least two steps to include distinct endpoints"
        )
    if cosine_first_step == final_step and minimum != base:
        raise ValueError(
            "cosine needs at least two steps to include distinct endpoints"
        )

    if step <= warmup_final_step:
        if step == 1:
            return warmup_start
        if step == warmup_final_step:
            return base
        progress = (step - 1) / (warmup_final_step - 1)
        return warmup_start + (base - warmup_start) * progress

    if step < cosine_first_step:
        return base
    if step == cosine_first_step:
        return base
    if step == final_step:
        return minimum
    progress = (step - cosine_first_step) / (
        final_step - cosine_first_step
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + (base - minimum) * cosine


__all__ = [
    "CONSTANT_SCHEDULE",
    "WARMUP_HOLD_COSINE_SCHEDULE",
    "absolute_step_from_cursor",
    "absolute_step_from_resume_state",
    "canonicalize_pgd2_schedule_config",
    "pgd2_learning_rate_at_step",
]

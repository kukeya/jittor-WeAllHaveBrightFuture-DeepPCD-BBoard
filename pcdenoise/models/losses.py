"""Pure-Jittor reconstruction losses used by the PGD profiles.

The fixed PGD commit and the AAAI paper do not implement the same InfoCD
formula.  They are deliberately exposed as separate functions so experiments
cannot silently mix their distance, normalization, or reduction semantics.
"""

from __future__ import annotations

import math
from numbers import Real

import jittor as jt

from pcdenoise.ops.indexing import _validate_batched_points
from pcdenoise.ops.knn import KNN_BACKENDS, knn


INFOCD_PROFILES = (
    "code_b5dc8be",
    "code_b5dc8be_sqdist",
    "paper",
)
_CODE_ALPHA = 0.5
_CODE_EPSILON = 1.0e-7
_CODE_MINIMUM_SQUARED_DISTANCE = 1.0e-9


def _positive_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return converted


def _validate_pair(
    prediction: jt.Var,
    target: jt.Var,
    *,
    validate_finite: bool,
) -> tuple[jt.Var, jt.Var]:
    if not isinstance(validate_finite, bool):
        raise ValueError("validate_finite must be a bool")
    prediction_points = _validate_batched_points(
        prediction,
        "prediction",
        require_finite=validate_finite,
    )
    target_points = _validate_batched_points(
        target,
        "target",
        require_finite=validate_finite,
    )
    if prediction_points.shape[0] != target_points.shape[0]:
        raise ValueError("prediction and target batch dimensions must match")
    if prediction_points.shape[2] != 3 or target_points.shape[2] != 3:
        raise ValueError("prediction and target must contain 3D points")
    if str(prediction_points.dtype) != str(target_points.dtype):
        raise ValueError("prediction and target dtypes must match")
    return prediction_points, target_points


def _directed_nearest_squared(
    source: jt.Var,
    reference: jt.Var,
    *,
    knn_backend: str,
    query_block_size: int,
    reference_block_size: int,
) -> jt.Var:
    squared, _ = knn(
        source,
        reference,
        1,
        backend=knn_backend,
        query_block_size=query_block_size,
        reference_block_size=reference_block_size,
        validate_finite=False,
    )
    return squared.squeeze(-1)


def _stable_logsumexp(values: jt.Var, *, dim: int) -> jt.Var:
    maximum = values.max(dim=dim, keepdims=True)
    return maximum + jt.log(
        jt.exp(values - maximum).sum(dim=dim, keepdims=True)
    )


def paper_infocd(
    prediction: jt.Var,
    target: jt.Var,
    *,
    alpha: float,
    knn_backend: str = "auto",
    query_block_size: int = 256,
    reference_block_size: int = 1024,
    validate_finite: bool = True,
) -> jt.Var:
    """Return equations (8)-(9) from the AAAI-26 PGD paper.

    This profile uses squared nearest-neighbor distance, normalizes the
    exponentials over the source points of each directed term, averages over
    points, and then averages the two directions and batch.
    """

    scale = _positive_finite_real(alpha, name="alpha")
    prediction_points, target_points = _validate_pair(
        prediction,
        target,
        validate_finite=validate_finite,
    )
    prediction_squared = _directed_nearest_squared(
        prediction_points,
        target_points,
        knn_backend=knn_backend,
        query_block_size=query_block_size,
        reference_block_size=reference_block_size,
    )
    target_squared = _directed_nearest_squared(
        target_points,
        prediction_points,
        knn_backend=knn_backend,
        query_block_size=query_block_size,
        reference_block_size=reference_block_size,
    )

    directed_losses: list[jt.Var] = []
    for squared in (prediction_squared, target_squared):
        logits = -scale * squared
        negative_log_probability = (
            -logits + _stable_logsumexp(logits, dim=1)
        )
        directed_losses.append(
            negative_log_probability.mean(dim=1)
        )
    return (
        0.5 * (directed_losses[0] + directed_losses[1])
    ).mean()


def code_b5dc8be_infocd(
    prediction: jt.Var,
    target: jt.Var,
    *,
    alpha: float = _CODE_ALPHA,
    knn_backend: str = "auto",
    query_block_size: int = 256,
    reference_block_size: int = 1024,
    validate_finite: bool = True,
) -> jt.Var:
    """Reproduce ``models/InfoCD.py`` from PGD commit ``b5dc8be``.

    The source takes the square root of clamped Chamfer distances, raises the
    normalization denominator to ``1e-7``, sums over points, and only averages
    over batch and direction.  Those unusual operations are preserved here as
    the code-faithful baseline rather than silently corrected.
    """

    fixed_scale = _positive_finite_real(alpha, name="alpha")
    if fixed_scale != _CODE_ALPHA:
        raise ValueError(
            "alpha is fixed at 0.5 for the code_b5dc8be profile"
        )
    prediction_points, target_points = _validate_pair(
        prediction,
        target,
        validate_finite=validate_finite,
    )
    prediction_squared = _directed_nearest_squared(
        prediction_points,
        target_points,
        knn_backend=knn_backend,
        query_block_size=query_block_size,
        reference_block_size=reference_block_size,
    )
    target_squared = _directed_nearest_squared(
        target_points,
        prediction_points,
        knn_backend=knn_backend,
        query_block_size=query_block_size,
        reference_block_size=reference_block_size,
    )

    directed_totals: list[jt.Var] = []
    for squared in (prediction_squared, target_squared):
        minimum = (
            jt.ones_like(squared) * _CODE_MINIMUM_SQUARED_DISTANCE
        )
        distance = jt.sqrt(jt.maximum(squared, minimum))
        exponentials = jt.exp(-fixed_scale * distance)
        denominator = (
            exponentials + _CODE_EPSILON
        ).sum(dim=1, keepdims=True)
        negative_log_ratio = (
            fixed_scale * distance
            + _CODE_EPSILON * jt.log(denominator)
        )
        directed_totals.append(negative_log_ratio.sum(dim=1))
    return (
        0.5 * (directed_totals[0] + directed_totals[1])
    ).mean()


def code_b5dc8be_sqdist_infocd(
    prediction: jt.Var,
    target: jt.Var,
    *,
    alpha: float = _CODE_ALPHA,
    knn_backend: str = "auto",
    query_block_size: int = 256,
    reference_block_size: int = 1024,
    validate_finite: bool = True,
) -> jt.Var:
    """Apply the code-faithful InfoCD formula to squared distances.

    This single-variable profile keeps the fixed alpha, epsilon-modified
    normalization, point sum, and direction/batch reductions from
    :func:`code_b5dc8be_infocd`.  Only the square root and its floor are
    removed, so the KNN squared distances enter both formula terms directly.
    """

    fixed_scale = _positive_finite_real(alpha, name="alpha")
    if fixed_scale != _CODE_ALPHA:
        raise ValueError(
            "alpha is fixed at 0.5 for the code_b5dc8be_sqdist profile"
        )
    prediction_points, target_points = _validate_pair(
        prediction,
        target,
        validate_finite=validate_finite,
    )
    prediction_squared = _directed_nearest_squared(
        prediction_points,
        target_points,
        knn_backend=knn_backend,
        query_block_size=query_block_size,
        reference_block_size=reference_block_size,
    )
    target_squared = _directed_nearest_squared(
        target_points,
        prediction_points,
        knn_backend=knn_backend,
        query_block_size=query_block_size,
        reference_block_size=reference_block_size,
    )

    directed_totals: list[jt.Var] = []
    for squared in (prediction_squared, target_squared):
        exponentials = jt.exp(-fixed_scale * squared)
        denominator = (
            exponentials + _CODE_EPSILON
        ).sum(dim=1, keepdims=True)
        negative_log_ratio = (
            fixed_scale * squared
            + _CODE_EPSILON * jt.log(denominator)
        )
        directed_totals.append(negative_log_ratio.sum(dim=1))
    return (
        0.5 * (directed_totals[0] + directed_totals[1])
    ).mean()


def correspondence_huber_loss(
    prediction: jt.Var,
    target: jt.Var,
    *,
    delta: float = 0.005,
    validate_finite: bool = True,
) -> jt.Var:
    """Return radial Huber loss for known same-index point correspondences.

    The quadratic branch is divided by ``delta`` so both branches use length
    units and meet continuously with value ``0.5 * delta``.  A strictly
    positive floor is used only in the unselected square-root branch at zero,
    keeping the exact zero loss and its gradient finite.
    """

    transition = _positive_finite_real(delta, name="delta")
    prediction_points, target_points = _validate_pair(
        prediction,
        target,
        validate_finite=validate_finite,
    )
    if tuple(prediction_points.shape) != tuple(target_points.shape):
        raise ValueError(
            "prediction and target shapes must match for correspondence loss"
        )
    squared = (
        (prediction_points - target_points) ** 2
    ).sum(dim=-1)
    radius = jt.sqrt(
        jt.maximum(
            squared,
            jt.ones_like(squared) * 1.0e-24,
        )
    )
    quadratic = 0.5 * squared / transition
    linear = radius - 0.5 * transition
    return jt.where(
        squared <= transition * transition,
        quadratic,
        linear,
    ).mean()


def infocd_loss(
    prediction: jt.Var,
    target: jt.Var,
    *,
    profile: str,
    alpha: float | None = None,
    knn_backend: str = "auto",
    query_block_size: int = 256,
    reference_block_size: int = 1024,
    validate_finite: bool = True,
) -> jt.Var:
    """Route an explicitly named InfoCD experiment profile."""

    if not isinstance(profile, str) or profile not in INFOCD_PROFILES:
        raise ValueError(
            f"profile must be one of {INFOCD_PROFILES}, got {profile!r}"
        )
    if profile == "code_b5dc8be":
        if alpha is not None:
            raise ValueError(
                "alpha must be omitted for the fixed code_b5dc8be profile"
            )
        return code_b5dc8be_infocd(
            prediction,
            target,
            knn_backend=knn_backend,
            query_block_size=query_block_size,
            reference_block_size=reference_block_size,
            validate_finite=validate_finite,
        )
    if profile == "code_b5dc8be_sqdist":
        if alpha is not None:
            raise ValueError(
                "alpha must be omitted for the fixed "
                "code_b5dc8be_sqdist profile"
            )
        return code_b5dc8be_sqdist_infocd(
            prediction,
            target,
            knn_backend=knn_backend,
            query_block_size=query_block_size,
            reference_block_size=reference_block_size,
            validate_finite=validate_finite,
        )
    if alpha is None:
        raise ValueError("alpha is required for the paper profile")
    return paper_infocd(
        prediction,
        target,
        alpha=alpha,
        knn_backend=knn_backend,
        query_block_size=query_block_size,
        reference_block_size=reference_block_size,
        validate_finite=validate_finite,
    )

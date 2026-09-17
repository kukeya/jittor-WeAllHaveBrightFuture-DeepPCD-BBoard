"""Differentiable squared Chamfer distance built from blocked Jittor KNN."""

from __future__ import annotations

import jittor as jt

from .indexing import _validate_batched_points
from .knn import knn


def chamfer_distance(
    source: jt.Var,
    target: jt.Var,
    *,
    query_block_size: int = 256,
    reference_block_size: int = 1024,
    return_components: bool = False,
    validate_finite: bool = True,
) -> jt.Var | tuple[jt.Var, jt.Var, jt.Var]:
    """Return the batch-mean sum of both directed mean squared distances.

    Set ``validate_finite=False`` only after a batch-level finite-value
    preflight; both internal KNN calls then avoid redundant synchronizations.
    """

    if not isinstance(validate_finite, bool):
        raise ValueError("validate_finite must be a bool")
    source_points = _validate_batched_points(
        source, "source", require_finite=validate_finite
    )
    target_points = _validate_batched_points(
        target, "target", require_finite=validate_finite
    )
    if source_points.shape[0] != target_points.shape[0]:
        raise ValueError("source and target batch dimensions must match")

    source_squared, _ = knn(
        source_points,
        target_points,
        1,
        query_block_size=query_block_size,
        reference_block_size=reference_block_size,
        validate_finite=False,
    )
    target_squared, _ = knn(
        target_points,
        source_points,
        1,
        query_block_size=query_block_size,
        reference_block_size=reference_block_size,
        validate_finite=False,
    )
    source_to_target = source_squared.squeeze(-1).mean(dim=1)
    target_to_source = target_squared.squeeze(-1).mean(dim=1)
    loss = source_to_target.mean() + target_to_source.mean()
    if return_components:
        return loss, source_to_target, target_to_source
    return loss

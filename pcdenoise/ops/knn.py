"""Blocked pure-Jittor K-nearest-neighbor reference implementation."""

from __future__ import annotations

from numbers import Integral

import jittor as jt

from .indexing import (
    _index_points_unchecked,
    _validate_batched_points,
    _validate_knn_coordinate_range,
)


KNN_BACKENDS = ("auto", "reference", "cuda")
# Enabled after the synchronized RTX 4090 benchmark in
# artifacts/progress/task07_cuda_knn/benchmark_review_fix.json: all parity
# cases passed and every tested shape exceeded the declared 1.20x
# median-speedup gate.
AUTO_CUDA_ENABLED = True


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _pairwise_squared(
    query: jt.Var,
    reference: jt.Var,
) -> jt.Var:
    """Compute one bounded ``(B, Q_block, R_block)`` distance tile."""

    query_norm = (query * query).sum(dim=-1, keepdims=True)
    reference_norm = (reference * reference).sum(
        dim=-1, keepdims=True
    ).transpose(0, 2, 1)
    cross = jt.nn.bmm(query, reference.transpose(0, 2, 1))
    raw = query_norm + reference_norm - 2.0 * cross
    return jt.maximum(raw, jt.zeros_like(raw))


def knn(
    query: jt.Var,
    reference: jt.Var,
    k: int,
    *,
    query_block_size: int = 256,
    reference_block_size: int = 1024,
    backend: str = "auto",
    validate_finite: bool = True,
) -> tuple[jt.Var, jt.Var]:
    """Return differentiable squared distances and discrete neighbor indices.

    The reference backend is tiled over both query and reference points.  The
    optional fused CUDA backend directly emits ``(B, Q, K)`` candidates.  Both
    stop the discrete search indices and recompute selected point differences,
    so returned squared distances retain gradients to both inputs. Non-float32
    reference searches cast candidate selection to float32; project point
    coordinates are therefore normalized before this operator. The safe
    default rejects finite coordinate magnitudes that could overflow squared
    distance. Set ``validate_finite=False`` only after a caller has guaranteed
    both finite and normalized coordinates.
    """

    if not isinstance(validate_finite, bool):
        raise ValueError("validate_finite must be a bool")
    query_points = _validate_batched_points(
        query, "query", require_finite=validate_finite
    )
    reference_points = _validate_batched_points(
        reference, "reference", require_finite=validate_finite
    )
    if validate_finite:
        _validate_knn_coordinate_range(query_points, "query")
        _validate_knn_coordinate_range(reference_points, "reference")
    neighbor_count = _positive_integer(k, "k")
    query_tile = _positive_integer(query_block_size, "query_block_size")
    reference_tile = _positive_integer(
        reference_block_size, "reference_block_size"
    )
    if query_points.shape[0] != reference_points.shape[0]:
        raise ValueError("query and reference batch dimensions must match")
    if query_points.shape[2] != reference_points.shape[2]:
        raise ValueError("query and reference channel dimensions must match")
    if str(query_points.dtype) != str(reference_points.dtype):
        raise ValueError("query and reference dtypes must match")
    if neighbor_count > reference_points.shape[1]:
        raise ValueError("k must not exceed the reference point count")
    if not isinstance(backend, str) or backend not in KNN_BACKENDS:
        raise ValueError(
            f"backend must be one of {KNN_BACKENDS}, got {backend!r}"
        )

    selected_backend = backend
    if backend == "auto":
        can_use_fused_cuda = (
            AUTO_CUDA_ENABLED
            and jt.has_cuda
            and int(jt.flags.use_cuda)
            and str(query_points.dtype) == "float32"
            and str(reference_points.dtype) == "float32"
            and neighbor_count in (8, 16, 32)
        )
        selected_backend = "cuda" if can_use_fused_cuda else "reference"

    if selected_backend == "cuda":
        from .cuda_knn import fused_cuda_knn_search

        _, indices = fused_cuda_knn_search(
            query_points,
            reference_points,
            neighbor_count,
            validate_finite=False,
        )
        selected_reference = _index_points_unchecked(
            reference_points, indices
        )
        differences = query_points.unsqueeze(2) - selected_reference
        exact_squared = (differences * differences).sum(dim=-1)
        return exact_squared, indices

    query_count = int(query_points.shape[1])
    reference_count = int(reference_points.shape[1])
    search_query_points = (
        query_points
        if str(query_points.dtype) == "float32"
        else query_points.float32()
    )
    search_reference_points = (
        reference_points
        if str(reference_points.dtype) == "float32"
        else reference_points.float32()
    )
    output_distances: list[jt.Var] = []
    output_indices: list[jt.Var] = []

    for query_start in range(0, query_count, query_tile):
        query_stop = min(query_start + query_tile, query_count)
        query_chunk = query_points[:, query_start:query_stop, :]
        search_query_chunk = search_query_points[
            :, query_start:query_stop, :
        ]
        best_search_distances: jt.Var | None = None
        best_indices: jt.Var | None = None

        for reference_start in range(0, reference_count, reference_tile):
            reference_stop = min(
                reference_start + reference_tile, reference_count
            )
            reference_chunk = search_reference_points[
                :, reference_start:reference_stop, :
            ]
            # Neighbor search is discrete.  Do not retain every search tile's
            # backward graph; selected distances are recomputed exactly below.
            tile_distances = _pairwise_squared(
                search_query_chunk, reference_chunk
            ).stop_grad()
            tile_k = min(neighbor_count, reference_stop - reference_start)
            local_distances, local_indices = jt.topk(
                tile_distances,
                tile_k,
                dim=-1,
                largest=False,
                sorted=True,
            )
            local_indices = local_indices + reference_start

            if best_search_distances is None:
                best_search_distances = local_distances
                best_indices = local_indices
                continue

            candidate_distances = jt.concat(
                (best_search_distances, local_distances), dim=-1
            )
            candidate_indices = jt.concat(
                (best_indices, local_indices), dim=-1
            )
            merge_k = min(neighbor_count, candidate_distances.shape[-1])
            best_search_distances, merge_positions = jt.topk(
                candidate_distances,
                merge_k,
                dim=-1,
                largest=False,
                sorted=True,
            )
            best_indices = candidate_indices.gather(-1, merge_positions)

        if best_indices is None:
            raise RuntimeError("KNN reference search produced no candidates")
        if best_indices.shape[-1] != neighbor_count:
            raise RuntimeError("KNN reference search produced too few neighbors")

        best_indices = best_indices.int32().stop_grad()
        selected_reference = _index_points_unchecked(
            reference_points, best_indices
        )
        differences = query_chunk.unsqueeze(2) - selected_reference
        exact_squared = (differences * differences).sum(dim=-1)
        output_distances.append(exact_squared)
        output_indices.append(best_indices)

    return (
        jt.concat(output_distances, dim=1),
        jt.concat(output_indices, dim=1).int32().stop_grad(),
    )

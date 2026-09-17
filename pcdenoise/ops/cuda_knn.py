"""Fused float32 CUDA KNN search without a dense pairwise matrix."""

from __future__ import annotations

from numbers import Integral

import jittor as jt

from .indexing import (
    _validate_batched_points,
    _validate_knn_coordinate_range,
)


SUPPORTED_K = (8, 16, 32)
QUERY_THREADS = 128
REFERENCE_TILE = 128


_CUDA_HEADER = r"""
#include <math_constants.h>

template <int K, int REFERENCE_TILE_SIZE>
__global__ void pcdenoise_fused_knn_kernel(
    const float* __restrict__ query,
    const float* __restrict__ reference,
    float* __restrict__ output_distances,
    int* __restrict__ output_indices,
    int query_count,
    int reference_count) {
    __shared__ float reference_tile[REFERENCE_TILE_SIZE * 3];

    const int batch_index = static_cast<int>(blockIdx.y);
    const int query_index =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
        + static_cast<int>(threadIdx.x);
    const bool active = query_index < query_count;

    const int query_batch_offset = batch_index * query_count * 3;
    const int reference_batch_offset = batch_index * reference_count * 3;
    float query_x = 0.0f;
    float query_y = 0.0f;
    float query_z = 0.0f;
    if (active) {
        const int query_offset = query_batch_offset + query_index * 3;
        query_x = query[query_offset];
        query_y = query[query_offset + 1];
        query_z = query[query_offset + 2];
    }

    float best_distances[K];
    int best_indices[K];
#pragma unroll
    for (int neighbor = 0; neighbor < K; ++neighbor) {
        best_distances[neighbor] = CUDART_INF_F;
        best_indices[neighbor] = 0x7fffffff;
    }

    for (
        int reference_start = 0;
        reference_start < reference_count;
        reference_start += REFERENCE_TILE_SIZE) {
        const int load_index = static_cast<int>(threadIdx.x);
        const int global_reference_index = reference_start + load_index;
        if (
            load_index < REFERENCE_TILE_SIZE
            && global_reference_index < reference_count) {
            const int reference_offset =
                reference_batch_offset + global_reference_index * 3;
            reference_tile[load_index * 3] = reference[reference_offset];
            reference_tile[load_index * 3 + 1] =
                reference[reference_offset + 1];
            reference_tile[load_index * 3 + 2] =
                reference[reference_offset + 2];
        }
        __syncthreads();

        int tile_count = reference_count - reference_start;
        if (tile_count > REFERENCE_TILE_SIZE) {
            tile_count = REFERENCE_TILE_SIZE;
        }
        if (active) {
            for (int local_reference = 0;
                 local_reference < tile_count;
                 ++local_reference) {
                const float delta_x =
                    query_x - reference_tile[local_reference * 3];
                const float delta_y =
                    query_y - reference_tile[local_reference * 3 + 1];
                const float delta_z =
                    query_z - reference_tile[local_reference * 3 + 2];
                float squared_distance =
                    delta_x * delta_x
                    + delta_y * delta_y
                    + delta_z * delta_z;
                if (!(squared_distance == squared_distance)) {
                    squared_distance = CUDART_INF_F;
                }
                const int candidate_index =
                    reference_start + local_reference;

                const bool enters_top_k =
                    squared_distance < best_distances[K - 1]
                    || (
                        squared_distance == best_distances[K - 1]
                        && candidate_index < best_indices[K - 1]);
                if (!enters_top_k) {
                    continue;
                }

                int insertion = K - 1;
#pragma unroll
                for (int neighbor = K - 1; neighbor > 0; --neighbor) {
                    const bool precedes_previous =
                        squared_distance < best_distances[neighbor - 1]
                        || (
                            squared_distance
                                == best_distances[neighbor - 1]
                            && candidate_index
                                < best_indices[neighbor - 1]);
                    if (precedes_previous) {
                        best_distances[neighbor] =
                            best_distances[neighbor - 1];
                        best_indices[neighbor] =
                            best_indices[neighbor - 1];
                        insertion = neighbor - 1;
                    }
                }
                best_distances[insertion] = squared_distance;
                best_indices[insertion] = candidate_index;
            }
        }
        __syncthreads();
    }

    if (active) {
        const int output_offset =
            (batch_index * query_count + query_index) * K;
#pragma unroll
        for (int neighbor = 0; neighbor < K; ++neighbor) {
            output_distances[output_offset + neighbor] =
                best_distances[neighbor];
            output_indices[output_offset + neighbor] =
                best_indices[neighbor];
        }
    }
}
"""


def _validate_supported_k(k: int, reference_count: int) -> int:
    if isinstance(k, bool) or not isinstance(k, Integral):
        raise ValueError("CUDA KNN k must be 8, 16, or 32")
    neighbor_count = int(k)
    if neighbor_count not in SUPPORTED_K:
        raise ValueError("CUDA KNN k must be 8, 16, or 32")
    if neighbor_count > reference_count:
        raise ValueError("k must not exceed the reference point count")
    return neighbor_count


def fused_cuda_knn_search(
    query: jt.Var,
    reference: jt.Var,
    k: int,
    *,
    validate_finite: bool = True,
) -> tuple[jt.Var, jt.Var]:
    """Return CUDA search distances and indices with shape ``(B, Q, K)``.

    Each CUDA thread owns one query and maintains its sorted top-k in registers.
    A block cooperatively loads bounded reference tiles into shared memory, so
    no ``B x Q x R`` tensor is allocated. The safe default requires normalized
    coordinates and rejects finite magnitudes that could overflow float32
    squared distance. Disabling validation can return ``inf`` distances, but
    indices remain legal and deterministically ordered.
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
    if query_points.shape[0] != reference_points.shape[0]:
        raise ValueError("query and reference batch dimensions must match")
    if query_points.shape[2] != reference_points.shape[2]:
        raise ValueError("query and reference channel dimensions must match")
    neighbor_count = _validate_supported_k(
        k, int(reference_points.shape[1])
    )
    if (
        str(query_points.dtype) != "float32"
        or str(reference_points.dtype) != "float32"
    ):
        raise TypeError("fused CUDA KNN requires float32 inputs")
    if not jt.has_cuda or not int(jt.flags.use_cuda):
        raise RuntimeError(
            "fused CUDA KNN requires CUDA with jt.flags.use_cuda=1"
        )

    batch_size = int(query_points.shape[0])
    query_count = int(query_points.shape[1])
    reference_count = int(reference_points.shape[1])
    output_shapes = [
        (batch_size, query_count, neighbor_count),
        (batch_size, query_count, neighbor_count),
    ]
    cuda_source = f"""
        const int threads = {QUERY_THREADS};
        const dim3 block(threads, 1, 1);
        const dim3 grid(
            (in0_shape1 + threads - 1) / threads,
            in0_shape0,
            1);
        pcdenoise_fused_knn_kernel<
            {neighbor_count}, {REFERENCE_TILE}><<<grid, block>>>(
                in0_p,
                in1_p,
                out0_p,
                out1_p,
                in0_shape1,
                in1_shape1);
    """
    search_distances, indices = jt.code(
        output_shapes,
        ["float32", "int32"],
        [query_points, reference_points],
        cuda_header=_CUDA_HEADER,
        cuda_src=cuda_source,
    )
    return (
        search_distances.stop_grad(),
        indices.int32().stop_grad(),
    )


__all__ = [
    "QUERY_THREADS",
    "REFERENCE_TILE",
    "SUPPORTED_K",
    "fused_cuda_knn_search",
]

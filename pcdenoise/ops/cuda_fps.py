"""Fused float32 CUDA farthest-point sampling for normalized point clouds."""

from __future__ import annotations

from numbers import Integral

import jittor as jt

from .indexing import (
    _validate_batched_points,
    _validate_knn_coordinate_range,
)


FPS_THREADS = 256


_CUDA_HEADER = r"""
#include <math_constants.h>

template <int THREAD_COUNT>
__global__ void pcdenoise_fused_fps_kernel(
    const float* __restrict__ points,
    int* __restrict__ output_indices,
    float* __restrict__ minimum_distances,
    int point_count,
    int sample_count,
    int start_index) {
    __shared__ float thread_best_distances[THREAD_COUNT];
    __shared__ int thread_best_indices[THREAD_COUNT];
    __shared__ int current_index;

    const int batch_index = static_cast<int>(blockIdx.x);
    const int thread_index = static_cast<int>(threadIdx.x);
    const int point_batch_offset = batch_index * point_count * 3;
    const int distance_batch_offset = batch_index * point_count;
    const int output_batch_offset = batch_index * sample_count;

    for (
        int point_index = thread_index;
        point_index < point_count;
        point_index += THREAD_COUNT) {
        minimum_distances[distance_batch_offset + point_index] =
            CUDART_INF_F;
    }
    if (thread_index == 0) {
        current_index = start_index;
    }
    __syncthreads();

    for (int sample_index = 0;
         sample_index < sample_count;
         ++sample_index) {
        const int selected_index = current_index;
        if (thread_index == 0) {
            output_indices[output_batch_offset + sample_index] =
                selected_index;
        }

        const int selected_offset =
            point_batch_offset + selected_index * 3;
        const float selected_x = points[selected_offset];
        const float selected_y = points[selected_offset + 1];
        const float selected_z = points[selected_offset + 2];
        float local_best_distance = -1.0f;
        int local_best_index = 0x7fffffff;

        for (
            int point_index = thread_index;
            point_index < point_count;
            point_index += THREAD_COUNT) {
            const int point_offset =
                point_batch_offset + point_index * 3;
            const float delta_x = points[point_offset] - selected_x;
            const float delta_y = points[point_offset + 1] - selected_y;
            const float delta_z = points[point_offset + 2] - selected_z;
            float squared_distance =
                delta_x * delta_x
                + delta_y * delta_y
                + delta_z * delta_z;
            if (!(squared_distance == squared_distance)) {
                squared_distance = CUDART_INF_F;
            }
            const int distance_offset =
                distance_batch_offset + point_index;
            const float previous = minimum_distances[distance_offset];
            float updated =
                squared_distance < previous ? squared_distance : previous;
            if (point_index == selected_index) {
                // Excluding every selected index guarantees uniqueness even
                // for duplicate or fully identical point sets.
                updated = -1.0f;
            }
            minimum_distances[distance_offset] = updated;
            if (
                updated > local_best_distance
                || (
                    updated == local_best_distance
                    && point_index < local_best_index)) {
                local_best_distance = updated;
                local_best_index = point_index;
            }
        }

        thread_best_distances[thread_index] = local_best_distance;
        thread_best_indices[thread_index] = local_best_index;
        __syncthreads();

        for (int offset = THREAD_COUNT / 2;
             offset > 0;
             offset >>= 1) {
            if (thread_index < offset) {
                const float candidate_distance =
                    thread_best_distances[thread_index + offset];
                const int candidate_index =
                    thread_best_indices[thread_index + offset];
                const float incumbent_distance =
                    thread_best_distances[thread_index];
                const int incumbent_index =
                    thread_best_indices[thread_index];
                if (
                    candidate_distance > incumbent_distance
                    || (
                        candidate_distance == incumbent_distance
                        && candidate_index < incumbent_index)) {
                    thread_best_distances[thread_index] =
                        candidate_distance;
                    thread_best_indices[thread_index] =
                        candidate_index;
                }
            }
            __syncthreads();
        }
        if (thread_index == 0 && sample_index + 1 < sample_count) {
            current_index = thread_best_indices[0];
        }
        __syncthreads();
    }
}
"""


def _positive_count(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def fused_cuda_fps(
    points: jt.Var,
    sample_count: int,
    *,
    start_index: int = 0,
    validate_finite: bool = True,
) -> jt.Var:
    """Return unique batch-local FPS indices with shape ``(B,M)``.

    One CUDA block owns each batch.  Threads cooperatively update a bounded
    ``(B,N)`` minimum-distance scratch array and reduce the next farthest
    index using a deterministic distance/index total order.
    """

    if not isinstance(validate_finite, bool):
        raise ValueError("validate_finite must be a bool")
    coordinates = _validate_batched_points(
        points,
        "points",
        require_finite=validate_finite,
    )
    if validate_finite:
        _validate_knn_coordinate_range(coordinates, "points")
    count = _positive_count(sample_count, name="sample_count")
    point_count = int(coordinates.shape[1])
    if count > point_count:
        raise ValueError("sample_count must not exceed the point count")
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, Integral)
        or not 0 <= int(start_index) < point_count
    ):
        raise ValueError("start_index is out of range")
    if str(coordinates.dtype) != "float32":
        raise TypeError("fused CUDA FPS requires float32 inputs")
    if not jt.has_cuda or not int(jt.flags.use_cuda):
        raise RuntimeError("fused CUDA FPS requires CUDA")

    batch_size = int(coordinates.shape[0])
    cuda_source = f"""
        const dim3 block({FPS_THREADS}, 1, 1);
        const dim3 grid(in0_shape0, 1, 1);
        pcdenoise_fused_fps_kernel<{FPS_THREADS}><<<grid, block>>>(
            in0_p,
            out0_p,
            out1_p,
            in0_shape1,
            {count},
            {int(start_index)});
    """
    indices, _ = jt.code(
        [
            (batch_size, count),
            (batch_size, point_count),
        ],
        ["int32", "float32"],
        [coordinates],
        cuda_header=_CUDA_HEADER,
        cuda_src=cuda_source,
    )
    return indices.int32().stop_grad()


__all__ = ["FPS_THREADS", "fused_cuda_fps"]

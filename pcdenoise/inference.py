"""Code-faithful whole-cloud patch inference for pure-Jittor denoisers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

import jittor as jt
import numpy as np
from scipy.spatial import cKDTree

from pcdenoise.data.mesh_dataset import UnitSphereTransform, fit_unit_sphere


NORMALIZATION_MODES = ("noisy_max", "identity", "robust_quantile")
FUSION_MODES = ("hard_best",)
_NOISE_ROUTE_COVERAGE_NUMERATOR = 99
_NOISE_ROUTE_COVERAGE_DENOMINATOR = 100
NOISE_ROUTE_MINIMUM_COVERAGE_RATIO = (
    _NOISE_ROUTE_COVERAGE_NUMERATOR
    / _NOISE_ROUTE_COVERAGE_DENOMINATOR
)


def _points(values: object, *, name: str) -> np.ndarray:
    points = np.asarray(values)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        raise ValueError(f"{name} must have finite shape (N, 3), N > 0")
    if not np.issubdtype(points.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    result = np.ascontiguousarray(points, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return converted


def _unit_interval_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in (0, 1]")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 < converted <= 1.0:
        raise ValueError(f"{name} must be a finite number in (0, 1]")
    return converted


def _mode(
    value: object,
    *,
    name: str,
    choices: tuple[str, ...],
) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(
            f"{name} must be one of {', '.join(repr(item) for item in choices)}"
        )
    return value


def _strict_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def canonical_inference_options(
    *,
    normalization_mode: object = "noisy_max",
    robust_quantile: object = None,
    fusion_mode: object = "hard_best",
    iteration_damping: object = 1.0,
) -> dict[str, object]:
    """Validate and canonicalize inference-only ablation options.

    ``robust_quantile`` is deliberately inactive unless
    ``normalization_mode='robust_quantile'``.  Requiring it explicitly avoids
    an unrecorded calibration default in test inference.
    """

    normalization = _mode(
        normalization_mode,
        name="normalization_mode",
        choices=NORMALIZATION_MODES,
    )
    fusion = _mode(
        fusion_mode,
        name="fusion_mode",
        choices=FUSION_MODES,
    )
    damping = _unit_interval_real(
        iteration_damping,
        name="iteration_damping",
    )
    if normalization == "robust_quantile":
        if robust_quantile is None:
            raise ValueError(
                "robust_quantile is required when "
                "normalization_mode='robust_quantile'"
            )
        quantile: float | None = _unit_interval_real(
            robust_quantile,
            name="robust_quantile",
        )
    else:
        if robust_quantile is not None:
            raise ValueError(
                "robust_quantile is only valid when "
                "normalization_mode='robust_quantile'"
            )
        quantile = None
    return {
        "normalization_mode": normalization,
        "robust_quantile": quantile,
        "fusion_mode": fusion,
        "iteration_damping": damping,
    }


def _normalization_transform(
    noisy_points: np.ndarray,
    *,
    normalization_mode: str,
    robust_quantile: float | None,
) -> UnitSphereTransform:
    if normalization_mode == "noisy_max":
        return fit_unit_sphere(noisy_points)
    if normalization_mode == "identity":
        return UnitSphereTransform(
            center=np.zeros(3, dtype=np.float64),
            scale=1.0,
        )
    if normalization_mode != "robust_quantile" or robust_quantile is None:
        raise RuntimeError("canonical normalization options became invalid")

    # This transform is fitted independently to this sample's noisy cloud.
    # It never observes a clean reference, another test sample, or dataset-wide
    # statistics.  NumPy's explicit linear order-statistic interpolation makes
    # the CPU result deterministic.
    coordinates = noisy_points.astype(np.float64)
    center = np.median(coordinates, axis=0)
    radii = np.linalg.norm(coordinates - center, axis=1)
    scale = float(
        np.quantile(
            radii,
            robust_quantile,
            method="linear",
        )
    )
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "noisy_points have degenerate robust_quantile normalization scale"
        )
    return UnitSphereTransform(center=center, scale=scale)


def numpy_farthest_point_indices(
    points: np.ndarray,
    sample_count: int,
    *,
    start_index: int = 0,
) -> np.ndarray:
    """Return deterministic FPS indices for one non-differentiable cloud."""

    cloud = _points(points, name="points")
    count = _positive_integer(sample_count, name="sample_count")
    point_count = len(cloud)
    if count > point_count:
        raise ValueError("sample_count must not exceed the point count")
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, Integral)
        or not 0 <= int(start_index) < point_count
    ):
        raise ValueError("start_index is out of range")

    coordinates = cloud.astype(np.float64)
    output = np.empty(count, dtype=np.int64)
    minimum_squared = np.full(point_count, np.inf, dtype=np.float64)
    selected = np.zeros(point_count, dtype=bool)
    current = int(start_index)
    for step in range(count):
        output[step] = current
        selected[current] = True
        delta = coordinates - coordinates[current]
        squared = np.einsum("ij,ij->i", delta, delta)
        minimum_squared = np.minimum(minimum_squared, squared)
        if step + 1 < count:
            scores = np.where(selected, -1.0, minimum_squared)
            # np.argmax returns the lower index for an exact distance tie.
            current = int(np.argmax(scores))
    return output


@dataclass(frozen=True)
class PatchPlan:
    """Immutable neighborhood and hard-best fallback plan."""

    seed_indices: np.ndarray
    indices: np.ndarray
    normalized_squared_distances: np.ndarray
    owner_patch: np.ndarray
    owner_local_index: np.ndarray
    uncovered_count: int

    @property
    def num_patches(self) -> int:
        return int(self.indices.shape[0])


def build_patch_plan(
    points: np.ndarray,
    *,
    patch_size: int = 1000,
    seed_k: float = 6,
    fps_start_index: int = 0,
) -> PatchPlan:
    """Build PGD FPS/KNN patches and its code-faithful hard-best fusion map."""

    cloud = _points(points, name="points")
    count = _positive_integer(patch_size, name="patch_size")
    density = _positive_real(seed_k, name="seed_k")
    point_count = len(cloud)
    if count > point_count:
        raise ValueError("patch_size must not exceed the point count")
    num_patches = int(density * point_count / count)
    if num_patches <= 0:
        raise ValueError(
            "seed_k * point_count / patch_size must produce at least one patch"
        )
    if num_patches > point_count:
        raise ValueError("seed_k produces more seed patches than points")

    seed_indices = numpy_farthest_point_indices(
        cloud,
        num_patches,
        start_index=fps_start_index,
    )
    tree = cKDTree(cloud.astype(np.float64), compact_nodes=True)
    distances, indices = tree.query(
        cloud[seed_indices].astype(np.float64),
        k=count,
        workers=1,
    )
    distances = np.asarray(distances, dtype=np.float64).reshape(
        num_patches, count
    )
    indices = np.asarray(indices, dtype=np.int64).reshape(
        num_patches, count
    )

    # cKDTree orders by distance but does not promise an index tie break.
    # Canonicalizing every returned row makes ordinary ties deterministic.
    for patch_index in range(num_patches):
        order = np.lexsort(
            (indices[patch_index], distances[patch_index])
        )
        indices[patch_index] = indices[patch_index, order]
        distances[patch_index] = distances[patch_index, order]

    squared = distances * distances
    denominator = squared[:, -1:]
    denominator = np.maximum(
        denominator,
        np.finfo(np.float64).tiny,
    )
    normalized_squared = squared / denominator
    if not np.isfinite(normalized_squared).all():
        raise RuntimeError("patch distance normalization produced non-finite values")

    best_distance = np.full(point_count, np.inf, dtype=np.float64)
    owner_patch = np.full(point_count, -1, dtype=np.int64)
    owner_local = np.full(point_count, -1, dtype=np.int64)
    local_indices = np.arange(count, dtype=np.int64)
    for patch_index in range(num_patches):
        point_indices = indices[patch_index]
        candidates = normalized_squared[patch_index]
        better = candidates < best_distance[point_indices]
        accepted_points = point_indices[better]
        best_distance[accepted_points] = candidates[better]
        owner_patch[accepted_points] = patch_index
        owner_local[accepted_points] = local_indices[better]

    uncovered = int((owner_patch < 0).sum())
    return PatchPlan(
        seed_indices=np.ascontiguousarray(seed_indices),
        indices=np.ascontiguousarray(indices),
        normalized_squared_distances=np.ascontiguousarray(
            normalized_squared.astype(np.float32)
        ),
        owner_patch=np.ascontiguousarray(owner_patch),
        owner_local_index=np.ascontiguousarray(owner_local),
        uncovered_count=uncovered,
    )


def _owner_weighted_noise_summary(
    patch_scales: np.ndarray,
    owner_counts: np.ndarray,
    *,
    conditioning_maximum_scale: float,
    uncovered_count: int,
) -> dict[str, object]:
    """Summarize per-patch estimates with one vote per owned input point."""

    scales = np.asarray(patch_scales, dtype=np.float64)
    counts = np.asarray(owner_counts, dtype=np.int64)
    if scales.ndim != 1 or counts.ndim != 1 or scales.shape != counts.shape:
        raise ValueError("patch scales and owner counts must be aligned vectors")
    if len(scales) == 0 or not np.isfinite(scales).all():
        raise ValueError("patch scales must be a nonempty finite vector")
    if (counts < 0).any():
        raise ValueError("owner counts must be nonnegative")
    if (
        isinstance(uncovered_count, bool)
        or not isinstance(uncovered_count, Integral)
        or int(uncovered_count) < 0
    ):
        raise ValueError("uncovered_count must be a nonnegative integer")
    uncovered = int(uncovered_count)
    maximum = _positive_real(
        conditioning_maximum_scale,
        name="conditioning_maximum_scale",
    )
    owned_point_count = int(counts.sum())
    if owned_point_count <= 0:
        raise RuntimeError("noise route capture has no owned input points")
    point_count = owned_point_count + uncovered
    # The frozen 0.99 threshold is exactly 99/100.  Integer comparison keeps
    # the boundary deterministic and accepts coverage equal to 0.99.
    if (
        owned_point_count * _NOISE_ROUTE_COVERAGE_DENOMINATOR
        < point_count * _NOISE_ROUTE_COVERAGE_NUMERATOR
    ):
        raise RuntimeError(
            "noise route coverage ratio is below the frozen minimum 0.99"
        )
    coverage_ratio = owned_point_count / point_count

    # Owner counts are integers, so materializing one scalar per input point is
    # exact.  inverted_cdf is the deterministic weighted nearest-rank rule:
    # the selected scale is the first whose cumulative owner count reaches the
    # requested fraction of the cloud.
    per_point = np.repeat(scales, counts)
    quantiles = np.quantile(
        per_point,
        (0.5, 0.75, 0.9),
        method="inverted_cdf",
    )
    q50, q75, q90 = (float(value) for value in quantiles)
    weighted_mad = float(
        np.quantile(
            np.abs(per_point - q50),
            0.5,
            method="inverted_cdf",
        )
    )
    # The model estimates float32 scales.  Compare against the representable
    # float32 form of its configured upper endpoint, rather than a slightly
    # different Python-float spelling of the same endpoint.
    represented_maximum = float(np.float32(maximum))
    saturated_points = int(counts[scales >= represented_maximum].sum())
    return {
        "aggregation": "hard_best_owner_patch_point_weighted_v1",
        "route_score_name": "q75",
        "route_score": q75,
        "q50": q50,
        "q75": q75,
        "q90": q90,
        "weighted_mad": weighted_mad,
        "conditioning_maximum_scale": maximum,
        "conditioning_upper_saturation_fraction": (
            saturated_points / owned_point_count
        ),
        "owned_point_count": owned_point_count,
        "zero_owner_patch_count": int((counts == 0).sum()),
        "uncovered_count": uncovered,
        "coverage_ratio": coverage_ratio,
        "patches": [
            {
                "patch_index": patch_index,
                "estimated_noise_scale": float(scale),
                "owner_point_count": int(owner_count),
            }
            for patch_index, (scale, owner_count) in enumerate(
                zip(scales, counts)
            )
        ],
    }


def denoise_normalized_cloud(
    model: object,
    points: np.ndarray,
    *,
    patch_size: int = 1000,
    seed_k: float = 6,
    patch_batch_size: int = 5,
    fusion_mode: str = "hard_best",
    fps_start_index: int = 0,
    capture_noise_route: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Denoise one normalized cloud and fuse patch displacements in point order."""

    cloud = _points(points, name="points")
    batch_size = _positive_integer(
        patch_batch_size,
        name="patch_batch_size",
    )
    fusion = _mode(
        fusion_mode,
        name="fusion_mode",
        choices=FUSION_MODES,
    )
    capture_route = _strict_bool(
        capture_noise_route,
        name="capture_noise_route",
    )
    if capture_route and fusion != "hard_best":
        raise ValueError(
            "capture_noise_route requires fusion_mode='hard_best'"
        )
    conditioning_maximum_scale = None
    if capture_route:
        conditioning_maximum_scale = _positive_real(
            getattr(model, "conditioning_maximum_scale", None),
            name="model.conditioning_maximum_scale",
        )
    plan = build_patch_plan(
        cloud,
        patch_size=patch_size,
        seed_k=seed_k,
        fps_start_index=fps_start_index,
    )
    if capture_route:
        actual_uncovered = int((plan.owner_patch < 0).sum())
        if plan.uncovered_count != actual_uncovered:
            raise RuntimeError(
                "patch plan uncovered_count differs from owner_patch"
            )
        covered_count = len(cloud) - actual_uncovered
        if (
            covered_count * _NOISE_ROUTE_COVERAGE_DENOMINATOR
            < len(cloud) * _NOISE_ROUTE_COVERAGE_NUMERATOR
        ):
            raise RuntimeError(
                "noise route coverage ratio is below the frozen minimum 0.99"
            )
    centers = cloud[plan.seed_indices]
    centered_patches = (
        cloud[plan.indices] - centers[:, None, :]
    ).astype(np.float32)
    displacements = np.empty_like(centered_patches)
    captured_patch_scales: list[np.ndarray] = []

    eval_method = getattr(model, "eval", None)
    if not callable(eval_method):
        raise TypeError("model must provide eval() and be callable")
    is_training_method = getattr(model, "is_training", None)
    was_training = (
        bool(is_training_method())
        if callable(is_training_method)
        else False
    )
    eval_method()
    try:
        with jt.no_grad():
            for start in range(0, plan.num_patches, batch_size):
                stop = min(start + batch_size, plan.num_patches)
                inputs = jt.array(centered_patches[start:stop])
                if capture_route:
                    result = model(inputs, return_details=True)
                    if not isinstance(result, tuple) or len(result) != 2:
                        raise TypeError(
                            "model must return (prediction, details) when "
                            "capture_noise_route is enabled"
                        )
                    predictions, model_details = result
                    if not isinstance(model_details, Mapping):
                        raise TypeError("model details must be a mapping")
                    if "estimated_noise_scale" not in model_details:
                        raise ValueError(
                            "model details must contain estimated_noise_scale"
                        )
                    estimated_noise_scale = model_details[
                        "estimated_noise_scale"
                    ]
                    if not isinstance(estimated_noise_scale, jt.Var):
                        raise TypeError(
                            "estimated_noise_scale must be one Jittor Var"
                        )
                    expected_scale_shape = (stop - start, 1)
                    actual_scale_shape = tuple(
                        int(dimension)
                        for dimension in estimated_noise_scale.shape
                    )
                    if actual_scale_shape != expected_scale_shape:
                        raise ValueError(
                            "estimated_noise_scale shape "
                            f"{actual_scale_shape} does not match "
                            f"{expected_scale_shape}"
                        )
                    if model_details.get("conditioning_source") != "estimated":
                        raise ValueError(
                            "conditioning_source must be 'estimated'"
                        )
                    scale_values = np.asarray(
                        estimated_noise_scale.numpy(),
                        dtype=np.float32,
                    )
                    if not np.isfinite(scale_values).all():
                        raise RuntimeError(
                            "estimated_noise_scale must contain only finite values"
                        )
                    captured_patch_scales.append(
                        np.ascontiguousarray(scale_values[:, 0])
                    )
                else:
                    predictions = model(inputs)
                if not isinstance(predictions, jt.Var):
                    raise TypeError("model must return one Jittor Var")
                actual = np.asarray(predictions.numpy(), dtype=np.float32)
                expected_shape = (stop - start, int(patch_size), 3)
                if actual.shape != expected_shape:
                    raise ValueError(
                        f"model output shape {actual.shape} does not match "
                        f"{expected_shape}"
                    )
                if not np.isfinite(actual).all():
                    raise RuntimeError("model produced non-finite predictions")
                displacements[start:stop] = (
                    actual - centered_patches[start:stop]
                )
    finally:
        if was_training:
            train_method = getattr(model, "train", None)
            if callable(train_method):
                train_method()

    covered = plan.owner_patch >= 0
    covered_ids = np.flatnonzero(covered)
    output = cloud.copy()
    zero_weight_fallback_count = 0
    if len(covered_ids):
        chosen_displacement = displacements[
            plan.owner_patch[covered_ids],
            plan.owner_local_index[covered_ids],
        ]
        output[covered_ids] = (
            cloud[covered_ids] + chosen_displacement
        ).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("patch displacement fusion produced non-finite output")
    inference_details: dict[str, object] = {
        "point_count": len(cloud),
        "patch_size": int(patch_size),
        "seed_k": float(seed_k),
        "fps_start_index": int(fps_start_index),
        "num_patches": plan.num_patches,
        "patch_batch_size": batch_size,
        "fusion_mode": fusion,
        "zero_weight_fallback_count": zero_weight_fallback_count,
        "uncovered_count": plan.uncovered_count,
        "coverage_fraction": float(covered.mean()),
    }
    if capture_route:
        patch_scales = np.concatenate(captured_patch_scales).astype(
            np.float64,
            copy=False,
        )
        if patch_scales.shape != (plan.num_patches,):
            raise RuntimeError(
                "captured noise scales do not match the patch plan"
            )
        owner_counts = np.bincount(
            plan.owner_patch[plan.owner_patch >= 0],
            minlength=plan.num_patches,
        )
        assert conditioning_maximum_scale is not None
        inference_details["noise_route_diagnostic"] = (
            _owner_weighted_noise_summary(
                patch_scales,
                owner_counts,
                conditioning_maximum_scale=conditioning_maximum_scale,
                uncovered_count=plan.uncovered_count,
            )
        )
    return np.ascontiguousarray(output), inference_details


def denoise_cloud(
    model: object,
    noisy_points: np.ndarray,
    *,
    patch_size: int = 1000,
    seed_k: float = 6,
    patch_batch_size: int = 5,
    niters: int = 1,
    normalization_mode: str = "noisy_max",
    robust_quantile: float | None = None,
    fusion_mode: str = "hard_best",
    iteration_damping: float = 1.0,
    fps_start_index: int = 0,
    capture_noise_route: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit one selected noisy transform, run damped passes, and restore units."""

    noisy = _points(noisy_points, name="noisy_points")
    iteration_count = _positive_integer(niters, name="niters")
    capture_route = _strict_bool(
        capture_noise_route,
        name="capture_noise_route",
    )
    if capture_route and iteration_count != 1:
        raise ValueError("capture_noise_route currently requires niters=1")
    options = canonical_inference_options(
        normalization_mode=normalization_mode,
        robust_quantile=robust_quantile,
        fusion_mode=fusion_mode,
        iteration_damping=iteration_damping,
    )
    transform = _normalization_transform(
        noisy,
        normalization_mode=str(options["normalization_mode"]),
        robust_quantile=options["robust_quantile"],
    )
    current = transform.apply(noisy)
    iteration_details: list[dict[str, object]] = []
    route_diagnostic = None
    for iteration in range(iteration_count):
        candidate, details = denoise_normalized_cloud(
            model,
            current,
            patch_size=patch_size,
            seed_k=seed_k,
            patch_batch_size=patch_batch_size,
            fusion_mode=str(options["fusion_mode"]),
            fps_start_index=fps_start_index,
            capture_noise_route=capture_route,
        )
        if capture_route:
            route_diagnostic = details.pop("noise_route_diagnostic")
        applied_damping = (
            1.0
            if iteration == 0
            else float(options["iteration_damping"])
        )
        if iteration == 0 or applied_damping == 1.0:
            # Preserve the original full-pass assignment bit for bit.
            current = candidate
        else:
            current = (
                current
                + np.float32(applied_damping) * (candidate - current)
            ).astype(np.float32)
        details["iteration"] = iteration + 1
        details["residual_damping"] = applied_damping
        iteration_details.append(details)
    restored = transform.restore(current)
    cloud_details: dict[str, object] = {
        "normalization_mode": options["normalization_mode"],
        "robust_quantile": options["robust_quantile"],
        "normalization_center": transform.center.tolist(),
        "normalization_scale": float(transform.scale),
        "fusion_mode": options["fusion_mode"],
        "fps_start_index": int(fps_start_index),
        "iteration_damping": options["iteration_damping"],
        "niters": iteration_count,
        "iterations": iteration_details,
    }
    if capture_route:
        if route_diagnostic is None:
            raise RuntimeError("noise route diagnostic was not captured")
        cloud_details["noise_route_diagnostic"] = route_diagnostic
    return restored, cloud_details


__all__ = [
    "PatchPlan",
    "FUSION_MODES",
    "NORMALIZATION_MODES",
    "NOISE_ROUTE_MINIMUM_COVERAGE_RATIO",
    "build_patch_plan",
    "canonical_inference_options",
    "denoise_cloud",
    "denoise_normalized_cloud",
    "numpy_farthest_point_indices",
]

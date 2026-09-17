"""Cosine-soft vector quantization with deferred EMA state updates."""

from __future__ import annotations

import math
from typing import Mapping

import jittor as jt
import numpy as np
from jittor import nn


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _initial_codebook_array(
    value: object,
    *,
    codebook_size: int,
    feature_dim: int,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (codebook_size, feature_dim):
        raise ValueError(
            "initial_codebook must have shape "
            f"({codebook_size}, {feature_dim})"
        )
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError("initial_codebook must be real numeric")
    array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("initial_codebook must contain only finite values")
    if (np.linalg.norm(array, axis=1) <= 1e-12).any():
        raise ValueError("initial_codebook rows must have nonzero norm")
    return np.ascontiguousarray(array)


def _assignment_feature_sum(
    flat_weights: jt.Var,
    flat_features: jt.Var,
) -> jt.Var:
    """Return ``flat_weights.T @ flat_features`` without a broadcast product.

    Jittor 1.3.10 implements a two-dimensional ``jt.matmul`` as a broadcast,
    multiply, and reduction expression.  When that expression remains lazy
    inside the full VQ forward graph, fusion can retain a very large temporary
    until the deferred EMA update is synchronized.  The framework's external
    GEMM operators are both mathematically equivalent and a fusion boundary.

    EMA state is maintained in float32, so mixed-precision assignment/features
    are accumulated in float32 as well.  CUDA fails closed if Jittor was built
    without its cuBLAS operator rather than silently restoring the high-memory
    broadcast expression.
    """

    weight_shape = tuple(int(dimension) for dimension in flat_weights.shape)
    feature_shape = tuple(
        int(dimension) for dimension in flat_features.shape
    )
    if (
        len(weight_shape) != 2
        or len(feature_shape) != 2
        or weight_shape[0] != feature_shape[0]
        or weight_shape[0] <= 0
        or weight_shape[1] <= 0
        or feature_shape[1] <= 0
    ):
        raise ValueError(
            "assignment weights and features must be nonempty 2D arrays "
            "with the same row count"
        )

    weights = flat_weights.float32()
    features = flat_features.float32()
    if int(jt.flags.use_cuda):
        cublas_ops = getattr(jt.compile_extern, "cublas_ops", None)
        cublas_matmul = (
            getattr(cublas_ops, "cublas_matmul", None)
            if cublas_ops is not None
            else None
        )
        if cublas_matmul is None:
            raise RuntimeError(
                "low-memory VQ EMA statistics require Jittor's cuBLAS "
                "matmul operator when CUDA is enabled"
            )
        return cublas_matmul(
            weights, features, True, False
        ).float32().stop_grad()

    mkl_ops = getattr(jt.compile_extern, "mkl_ops", None)
    mkl_matmul = (
        getattr(mkl_ops, "mkl_matmul", None)
        if mkl_ops is not None
        else None
    )
    if mkl_matmul is not None:
        return mkl_matmul(
            weights, features, True, False
        ).float32().stop_grad()

    # Portable CPU fallback with O(K*D) output storage.  This intentionally
    # avoids jt.matmul, whose generic 2D implementation constructs K*M*D.
    output = jt.code(
        (weight_shape[1], feature_shape[1]),
        "float32",
        [weights, features],
        cpu_src="""
            for (int code_index = 0;
                 code_index < in0_shape1;
                 ++code_index) {
                for (int feature_index = 0;
                     feature_index < in1_shape1;
                     ++feature_index) {
                    float value = 0.0f;
                    for (int row_index = 0;
                         row_index < in0_shape0;
                         ++row_index) {
                        value += (
                            in0_p[
                                row_index * in0_shape1 + code_index]
                            * in1_p[
                                row_index * in1_shape1 + feature_index]);
                    }
                    out0_p[
                        code_index * out0_shape1 + feature_index] = value;
                }
            }
        """,
    )
    return output.stop_grad()


def _differentiable_matrix_product(
    left: jt.Var,
    right: jt.Var,
) -> jt.Var:
    """Multiply two 2D arrays with a differentiable CUDA GEMM path."""

    left_shape = tuple(int(dimension) for dimension in left.shape)
    right_shape = tuple(int(dimension) for dimension in right.shape)
    if (
        len(left_shape) != 2
        or len(right_shape) != 2
        or left_shape[1] != right_shape[0]
    ):
        raise ValueError("matrix-product inputs must be compatible 2D arrays")
    if not int(jt.flags.use_cuda):
        return jt.matmul(left, right)

    cublas_ops = getattr(jt.compile_extern, "cublas_ops", None)
    batched_matmul = (
        getattr(cublas_ops, "cublas_batched_matmul", None)
        if cublas_ops is not None
        else None
    )
    if batched_matmul is None:
        raise RuntimeError(
            "low-memory differentiable VQ matrix products require Jittor's "
            "cuBLAS batched matmul operator when CUDA is enabled"
        )

    # The source expression's output follows the left input dtype.  cuBLAS
    # requires both operands to have equal element width, so mirror that
    # promotion explicitly before adding a singleton batch dimension.  Unlike
    # Jittor's 2D cuBLAS operator, its batched operator defines exact backward
    # GEMMs for both inputs.
    target_dtype = str(left.dtype)
    typed_right = (
        right if str(right.dtype) == target_dtype else right.cast(target_dtype)
    )
    product = batched_matmul(
        left.reshape((1, left_shape[0], left_shape[1])),
        typed_right.reshape((1, right_shape[0], right_shape[1])),
        False,
        False,
    )
    return product.reshape((left_shape[0], right_shape[1]))


class SoftVectorQuantizer(nn.Module):
    """Soft codebook assignment with straight-through feature gradients.

    A training forward caches detached assignment sufficient statistics but
    leaves the current codebook unchanged. ``apply_pending_ema`` commits that
    update after the current backward/optimizer step. By default the next
    training forward performs that commit automatically before it computes new
    assignments. Evaluation forwards never cache or apply EMA state.
    """

    def __init__(
        self,
        feature_dim: int = 48,
        codebook_size: int = 128,
        *,
        temperature: float = 0.1,
        momentum: float = 0.99,
        epsilon: float = 1e-5,
        commitment_cost: float = 0.0,
        use_ema: bool = True,
        initial_codebook: object | None = None,
        auto_apply_pending: bool = True,
        dead_code_check_interval: int = 1000,
        dead_steps: int = 5000,
        dead_code_seed: int = 0,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_integer(
            feature_dim, name="feature_dim"
        )
        self.codebook_size = _positive_integer(
            codebook_size, name="codebook_size"
        )
        self.temperature = _finite_float(
            temperature, name="temperature"
        )
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.momentum = _finite_float(momentum, name="momentum")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        self.epsilon = _finite_float(epsilon, name="epsilon")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.commitment_cost = _finite_float(
            commitment_cost, name="commitment_cost"
        )
        if self.commitment_cost != 0.0:
            raise ValueError(
                "commitment_cost must remain zero for the source-faithful VQ"
            )
        if not isinstance(use_ema, bool):
            raise ValueError("use_ema must be a boolean")
        self.use_ema = use_ema
        if not isinstance(auto_apply_pending, bool):
            raise ValueError("auto_apply_pending must be a boolean")
        self.auto_apply_pending = auto_apply_pending
        self.dead_code_check_interval = _positive_integer(
            dead_code_check_interval, name="dead_code_check_interval"
        )
        self.dead_steps = _nonnegative_integer(
            dead_steps, name="dead_steps"
        )
        self.dead_code_seed = _nonnegative_integer(
            dead_code_seed, name="dead_code_seed"
        )

        if initial_codebook is None:
            codebook = jt.randn(self.codebook_size, self.feature_dim)
            codebook = self._normalize_rows(codebook)
        else:
            codebook = jt.array(
                _initial_codebook_array(
                    initial_codebook,
                    codebook_size=self.codebook_size,
                    feature_dim=self.feature_dim,
                )
            )
        self.codebook = codebook.float32().stop_grad()
        self.cluster_size = jt.zeros(
            (self.codebook_size,), dtype="float32"
        ).stop_grad()
        self.cluster_sum = jt.zeros(
            (self.codebook_size, self.feature_dim), dtype="float32"
        ).stop_grad()
        self.usage_count = jt.zeros(
            (self.codebook_size,), dtype="float32"
        ).stop_grad()
        self.last_usage = jt.zeros(
            (self.codebook_size,), dtype="int32"
        ).stop_grad()
        self.step_counter = jt.zeros((1,), dtype="int32").stop_grad()

        # Private graph references are intentionally absent from state_dict.
        self._pending_cluster_size: jt.Var | None = None
        self._pending_cluster_sum: jt.Var | None = None
        self._pending_quantized: jt.Var | None = None
        self._last_metrics: dict[str, jt.Var] | None = None

    @property
    def has_pending_ema(self) -> bool:
        """Whether one training batch is waiting for its EMA commit."""

        return self._pending_cluster_size is not None

    def _normalize_rows(self, values: jt.Var) -> jt.Var:
        squared_norm = (values * values).sum(dim=1, keepdims=True)
        minimum = jt.ones_like(squared_norm) * 1e-24
        norm = jt.sqrt(jt.maximum(squared_norm, minimum))
        return values / norm

    def _validate_features(self, features: jt.Var) -> tuple[int, ...]:
        if not isinstance(features, jt.Var):
            raise TypeError("features must be a Jittor Var")
        shape = tuple(int(dimension) for dimension in features.shape)
        if (
            len(shape) < 2
            or shape[-1] != self.feature_dim
            or any(dimension <= 0 for dimension in shape[:-1])
        ):
            raise ValueError(
                "features must have nonempty shape (..., feature_dim)"
            )
        dtype = str(features.dtype)
        if not (dtype.startswith("float") or dtype.startswith("bfloat")):
            raise ValueError("features must have a floating-point dtype")
        return shape

    def _soft_quantize_flat(
        self, features: jt.Var
    ) -> tuple[jt.Var, jt.Var, jt.Var, jt.Var]:
        shape = self._validate_features(features)
        flat_features = features.reshape((-1, self.feature_dim))
        normalized_features = self._normalize_rows(flat_features)
        normalized_codebook = self._normalize_rows(self.codebook)
        similarity = _differentiable_matrix_product(
            normalized_features, normalized_codebook.transpose(1, 0)
        )
        weights = nn.softmax(similarity / self.temperature, dim=1)
        flat_quantized = _differentiable_matrix_product(
            weights, self.codebook
        )
        indices = jt.argmax(similarity, dim=1)[0].int32().stop_grad()
        return (
            flat_quantized.reshape(shape),
            indices.reshape(shape[:-1]),
            weights.reshape(shape[:-1] + (self.codebook_size,)),
            flat_features,
        )

    def soft_quantize(
        self, features: jt.Var
    ) -> tuple[jt.Var, jt.Var, jt.Var]:
        """Return soft mixture, hard diagnostic index, and soft weights."""

        quantized, indices, weights, _ = self._soft_quantize_flat(features)
        return quantized, indices, weights

    def _record_metrics(self, flat_weights: jt.Var) -> None:
        weight_sums = flat_weights.sum(dim=1)
        assignment_mass = flat_weights.sum(dim=0)
        probability = assignment_mass / assignment_mass.sum()
        safe_probability = jt.maximum(
            probability, jt.ones_like(probability) * self.epsilon
        )
        active = (assignment_mass > self.epsilon).float32()
        metric_expressions = {
            "weight_sum_mean": weight_sums.mean(),
            "weight_sum_max_error": jt.abs(weight_sums - 1.0).max(),
            "utilization": active.mean(),
            "perplexity": jt.exp(
                -(probability * jt.log(safe_probability)).sum()
            ),
            "dead_codes": (1.0 - active).sum(),
        }
        self._last_metrics = {
            name: (value * 1.0).stop_grad()
            for name, value in metric_expressions.items()
        }

    def metrics(self) -> Mapping[str, jt.Var]:
        """Return finite health metrics from the most recent assignment."""

        if self._last_metrics is None:
            raise RuntimeError("metrics are unavailable before the first forward")
        return dict(self._last_metrics)

    def _cache_pending(
        self,
        flat_features: jt.Var,
        flat_weights: jt.Var,
        quantized: jt.Var,
    ) -> None:
        if self.has_pending_ema:
            raise RuntimeError(
                "pending EMA statistics already exist; call "
                "apply_pending_ema() before another training forward"
            )
        detached_features = (flat_features * 1.0).stop_grad()
        detached_weights = (flat_weights * 1.0).stop_grad()
        self._pending_cluster_size = (
            detached_weights.sum(dim=0) * 1.0
        ).stop_grad()
        self._pending_cluster_sum = _assignment_feature_sum(
            detached_weights, detached_features
        )
        # Materializing this value before in-place state mutation preserves the
        # old-codebook value of a still-lazy output graph.
        self._pending_quantized = (quantized * 1.0).stop_grad()

    def _clear_pending(self) -> None:
        self._pending_cluster_size = None
        self._pending_cluster_sum = None
        self._pending_quantized = None

    def _deterministic_dead_code_noise(self, step: int) -> jt.Var:
        """Return stateless Jittor-native normal noise for one reset check."""

        positions = jt.arange(
            self.codebook_size * self.feature_dim
        ).float32().reshape((self.codebook_size, self.feature_dim))
        seed = float(self.dead_code_seed)
        current_step = float(step)
        phase = positions + seed * 0.754877666 + current_step * 0.569840296
        uniform_one_raw = (
            jt.sin(phase * 12.9898 + 78.233) * 43758.5453
        )
        uniform_two_raw = (
            jt.sin(phase * 39.3467 + 11.135) * 24634.6345
        )
        uniform_one = uniform_one_raw - jt.floor(uniform_one_raw)
        uniform_two = uniform_two_raw - jt.floor(uniform_two_raw)
        lower = jt.ones_like(uniform_one) * 1e-7
        upper = jt.ones_like(uniform_one) * (1.0 - 1e-7)
        uniform_one = jt.minimum(
            jt.maximum(uniform_one, lower), upper
        )
        return (
            jt.sqrt(-2.0 * jt.log(uniform_one))
            * jt.cos(2.0 * math.pi * uniform_two)
        ).float32().stop_grad()

    def _maybe_reset_dead_codes(
        self,
        *,
        next_cluster_size: jt.Var,
        next_cluster_sum: jt.Var,
        next_usage_count: jt.Var,
        next_last_usage: jt.Var,
        next_step: jt.Var,
        next_codebook: jt.Var,
    ) -> tuple[jt.Var, jt.Var, jt.Var, jt.Var, jt.Var]:
        """Apply the source reset rule at its sparse training checkpoints."""

        if not self.is_training() or not self.use_ema:
            return (
                next_cluster_size,
                next_cluster_sum,
                next_usage_count,
                next_last_usage,
                next_codebook,
            )
        step = int(next_step.item())
        if step % self.dead_code_check_interval != 0:
            return (
                next_cluster_size,
                next_cluster_sum,
                next_usage_count,
                next_last_usage,
                next_codebook,
            )

        dead_mask = (step - next_last_usage) > self.dead_steps
        dead_indices = np.flatnonzero(dead_mask.numpy()).tolist()
        if not dead_indices:
            return (
                next_cluster_size,
                next_cluster_sum,
                next_usage_count,
                next_last_usage,
                next_codebook,
            )

        # The source mutates dead rows sequentially. Keeping a row list retains
        # that behavior even when a later dead row copies an earlier dead row.
        _, most_used = jt.topk(
            next_usage_count,
            len(dead_indices),
            dim=0,
            largest=True,
            sorted=True,
        )
        source_indices = most_used.numpy().astype(np.int64).tolist()
        noise = self._deterministic_dead_code_noise(step)
        rows = [next_codebook[index] for index in range(self.codebook_size)]
        for reset_rank, (dead_index, source_index) in enumerate(
            zip(dead_indices, source_indices)
        ):
            candidate = (
                rows[source_index] + 0.1 * noise[reset_rank]
            ).reshape((1, self.feature_dim))
            rows[dead_index] = self._normalize_rows(candidate)[0]
        next_codebook = jt.stack(rows, dim=0).float32().stop_grad()

        retained = (1 - dead_mask.int32()).float32()
        next_cluster_size = (
            next_cluster_size * retained
        ).float32().stop_grad()
        next_cluster_sum = (
            next_cluster_sum
            * retained.reshape((self.codebook_size, 1))
        ).float32().stop_grad()
        next_usage_count = (
            next_usage_count * retained
        ).float32().stop_grad()
        next_last_usage = jt.where(
            dead_mask,
            jt.ones_like(next_last_usage) * step,
            next_last_usage,
        ).int32().stop_grad()
        return (
            next_cluster_size,
            next_cluster_sum,
            next_usage_count,
            next_last_usage,
            next_codebook,
        )

    def apply_pending_ema(self) -> bool:
        """Commit one cached training batch, returning whether work was done."""

        if not self.has_pending_ema:
            return False
        pending_size = self._pending_cluster_size
        pending_sum = self._pending_cluster_sum
        pending_quantized = self._pending_quantized
        if (
            pending_size is None
            or pending_sum is None
            or pending_quantized is None
        ):
            raise RuntimeError("incomplete pending EMA state")

        next_cluster_size = (
            self.momentum * self.cluster_size
            + (1.0 - self.momentum) * pending_size
        ).stop_grad()
        next_cluster_sum = (
            self.momentum * self.cluster_sum
            + (1.0 - self.momentum) * pending_sum
        ).stop_grad()
        next_usage_count = (
            self.usage_count + pending_size
        ).stop_grad()
        next_step = (self.step_counter + 1).int32().stop_grad()
        next_last_usage = jt.where(
            pending_size > 0.0,
            jt.ones_like(self.last_usage) * next_step,
            self.last_usage,
        ).int32().stop_grad()
        codebook_means = next_cluster_sum / (
            next_cluster_size.reshape((self.codebook_size, 1))
            + self.epsilon
        )
        active = (next_cluster_size > self.epsilon).reshape(
            (self.codebook_size, 1)
        )
        next_codebook = jt.where(
            active, codebook_means, self.codebook
        ).float32().stop_grad()

        next_values = [
            pending_quantized,
            next_cluster_size,
            next_cluster_sum,
            next_usage_count,
            next_last_usage,
            next_step,
            next_codebook,
        ]
        jt.sync(next_values)
        (
            next_cluster_size,
            next_cluster_sum,
            next_usage_count,
            next_last_usage,
            next_codebook,
        ) = self._maybe_reset_dead_codes(
            next_cluster_size=next_cluster_size,
            next_cluster_sum=next_cluster_sum,
            next_usage_count=next_usage_count,
            next_last_usage=next_last_usage,
            next_step=next_step,
            next_codebook=next_codebook,
        )
        jt.sync(
            [
                next_cluster_size,
                next_cluster_sum,
                next_usage_count,
                next_last_usage,
                next_codebook,
            ]
        )
        self.cluster_size.update(next_cluster_size)
        self.cluster_sum.update(next_cluster_sum)
        self.usage_count.update(next_usage_count)
        self.last_usage.update(next_last_usage)
        self.step_counter.update(next_step)
        self.codebook.update(next_codebook)
        jt.sync(
            [
                self.cluster_size,
                self.cluster_sum,
                self.usage_count,
                self.last_usage,
                self.step_counter,
                self.codebook,
            ]
        )
        self._clear_pending()
        return True

    def execute(
        self,
        features: jt.Var,
        calculate_commitment_loss: bool = False,
    ) -> tuple[jt.Var, jt.Var]:
        """Quantize one feature tensor and return a zero commitment scalar."""

        if not isinstance(calculate_commitment_loss, bool):
            raise ValueError("calculate_commitment_loss must be a boolean")
        if self.is_training() and self.use_ema and self.has_pending_ema:
            if not self.auto_apply_pending:
                raise RuntimeError(
                    "pending EMA statistics exist; call apply_pending_ema() "
                    "before another training forward"
                )
            self.apply_pending_ema()

        quantized, _, weights, flat_features = self._soft_quantize_flat(
            features
        )
        flat_weights = weights.reshape((-1, self.codebook_size))
        self._record_metrics(flat_weights)
        if self.is_training() and self.use_ema:
            self._cache_pending(flat_features, flat_weights, quantized)

        output = features + (quantized - features).stop_grad()
        commitment = jt.zeros((1,), dtype=features.dtype).sum().stop_grad()
        return output, commitment


# The source project names this component CodebookModule.
CodebookModule = SoftVectorQuantizer


__all__ = ["CodebookModule", "SoftVectorQuantizer"]

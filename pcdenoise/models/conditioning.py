"""Small pure-Jittor conditioning modules for PGD ablations."""

from __future__ import annotations

import math
from numbers import Integral, Real

import jittor as jt
from jittor import nn

from pcdenoise.ops.indexing import _validate_batched_points


_FLOAT_DTYPES = {"float16", "float32", "float64", "bfloat16"}


def _positive_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _validate_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _features(
    value: object,
    *,
    name: str,
    channels: int,
    require_finite: bool,
) -> jt.Var:
    features = _validate_batched_points(
        value,
        name,
        require_xyz=False,
        require_finite=require_finite,
    )
    if int(features.shape[2]) != channels:
        raise ValueError(f"{name} must have {channels} channels")
    return features


def _noise_scale(
    value: object,
    *,
    batch_size: int,
    require_finite: bool,
) -> jt.Var:
    if not isinstance(value, jt.Var):
        raise TypeError("noise_scale must be a jittor.Var")
    if value.ndim == 1:
        if int(value.shape[0]) != batch_size:
            raise ValueError(
                "noise_scale batch dimension must match features"
            )
        scale = value.reshape((batch_size, 1))
    elif (
        value.ndim == 2
        and int(value.shape[0]) == batch_size
        and int(value.shape[1]) == 1
    ):
        scale = value
    else:
        raise ValueError("noise_scale must have shape (B,) or (B, 1)")
    if str(scale.dtype) not in _FLOAT_DTYPES:
        raise ValueError("noise_scale must have a floating dtype")
    if require_finite:
        finite_scale = (
            scale if str(scale.dtype) == "float32" else scale.float32()
        )
        if not bool(jt.isfinite(finite_scale).all().item()):
            raise ValueError("noise_scale must be finite")
        if float(finite_scale.min().item()) <= 0.0:
            raise ValueError("noise_scale must be positive")
    return scale


def _zero_linear(linear: nn.Linear) -> None:
    linear.weight.update(jt.zeros_like(linear.weight))
    if linear.bias is not None:
        linear.bias.update(jt.zeros_like(linear.bias))


class NoiseScaleEstimator(nn.Module):
    """Estimate one bounded noise scale from mean-pooled bottleneck features."""

    def __init__(
        self,
        feature_dim: int = 162,
        hidden_dim: int = 32,
        minimum_scale: float = 0.005,
        maximum_scale: float = 0.020,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_integer(
            feature_dim,
            name="feature_dim",
        )
        self.hidden_dim = _positive_integer(
            hidden_dim,
            name="hidden_dim",
        )
        self.minimum_scale = _finite_real(
            minimum_scale,
            name="minimum_scale",
        )
        self.maximum_scale = _finite_real(
            maximum_scale,
            name="maximum_scale",
        )
        if self.minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be positive")
        if self.maximum_scale <= self.minimum_scale:
            raise ValueError(
                "maximum_scale must be greater than minimum_scale"
            )
        self.minimum_log_scale = math.log(self.minimum_scale)
        self.maximum_log_scale = math.log(self.maximum_scale)
        self.linear1 = nn.Linear(self.feature_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, 1)

    def execute(
        self,
        bottleneck_features: jt.Var,
        *,
        validate_finite: bool = True,
    ) -> jt.Var:
        check_finite = _validate_bool(
            validate_finite,
            name="validate_finite",
        )
        features = _features(
            bottleneck_features,
            name="bottleneck_features",
            channels=self.feature_dim,
            require_finite=check_finite,
        )
        pooled = features.mean(dim=1)
        hidden = nn.relu(self.linear1(pooled))
        fraction = jt.sigmoid(self.linear2(hidden))
        log_scale = (
            self.minimum_log_scale
            + fraction
            * (self.maximum_log_scale - self.minimum_log_scale)
        )
        return jt.exp(log_scale)


class ScalarFiLM(nn.Module):
    """Apply identity-initialized FiLM from one positive scale per patch."""

    def __init__(
        self,
        feature_dim: int = 64,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_integer(
            feature_dim,
            name="feature_dim",
        )
        self.embedding_dim = _positive_integer(
            embedding_dim,
            name="embedding_dim",
        )
        self.embedding = nn.Linear(1, self.embedding_dim)
        self.affine = nn.Linear(
            self.embedding_dim,
            2 * self.feature_dim,
        )
        _zero_linear(self.affine)

    def execute(
        self,
        features: jt.Var,
        noise_scale: jt.Var,
        *,
        validate_finite: bool = True,
    ) -> jt.Var:
        check_finite = _validate_bool(
            validate_finite,
            name="validate_finite",
        )
        values = _features(
            features,
            name="features",
            channels=self.feature_dim,
            require_finite=check_finite,
        )
        batch_size = int(values.shape[0])
        scale = _noise_scale(
            noise_scale,
            batch_size=batch_size,
            require_finite=check_finite,
        )
        embedded = nn.relu(self.embedding(jt.log(scale)))
        parameters = self.affine(embedded)
        gamma, beta = parameters.chunk(2, dim=-1)
        gamma = gamma.reshape((batch_size, 1, self.feature_dim))
        beta = beta.reshape((batch_size, 1, self.feature_dim))
        return values * (1.0 + gamma) + beta


class PointwiseDisplacementGate(nn.Module):
    """Predict an identity-initialized per-point displacement multiplier."""

    def __init__(
        self,
        feature_dim: int = 32,
        hidden_dim: int = 32,
        identity_probability: float = 0.95,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_integer(
            feature_dim,
            name="feature_dim",
        )
        self.hidden_dim = _positive_integer(
            hidden_dim,
            name="hidden_dim",
        )
        self.identity_probability = _finite_real(
            identity_probability,
            name="identity_probability",
        )
        if not 0.0 < self.identity_probability < 1.0:
            raise ValueError(
                "identity_probability must be strictly between zero and one"
            )
        self.identity_logit = math.log(
            self.identity_probability
            / (1.0 - self.identity_probability)
        )
        self.linear1 = nn.Linear(self.feature_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, 1)
        _zero_linear(self.linear2)

    def execute(
        self,
        point_features: jt.Var,
        *,
        validate_finite: bool = True,
    ) -> jt.Var:
        check_finite = _validate_bool(
            validate_finite,
            name="validate_finite",
        )
        features = _features(
            point_features,
            name="point_features",
            channels=self.feature_dim,
            require_finite=check_finite,
        )
        hidden = nn.relu(self.linear1(features))
        logits = self.linear2(hidden)
        numerator = jt.sigmoid(logits + self.identity_logit)
        denominator = jt.sigmoid(
            jt.zeros_like(logits) + self.identity_logit
        )
        return numerator / denominator


__all__ = [
    "NoiseScaleEstimator",
    "PointwiseDisplacementGate",
    "ScalarFiLM",
]

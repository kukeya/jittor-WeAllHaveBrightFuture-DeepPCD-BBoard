"""Four-level pure-Jittor PGD point-cloud denoiser."""

from __future__ import annotations

import math
import struct
from numbers import Integral, Real
from typing import Sequence

import jittor as jt
from jittor import nn

from pcdenoise.ops.indexing import _validate_batched_points

from .blocks import (
    Downsampling,
    StartBlock,
    Upsampling,
    official_pgd_shape_ledger,
    validate_decoder_reshape,
)
from .conditioning import (
    NoiseScaleEstimator,
    PointwiseDisplacementGate,
    ScalarFiLM,
)
from .vq import SoftVectorQuantizer


DEFAULT_FEATURE_DIMS = (32, 48, 72, 108, 162)
DEFAULT_CODEBOOK_SIZES = (512, 384, 256, 192)
DEFAULT_DOWNSAMPLE_STRIDES = (4, 3, 2, 1)
_FLOAT_DTYPES = {"float16", "float32", "float64", "bfloat16"}


def _positive_integer(value: object, *, name: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _integer_tuple(
    values: object,
    *,
    name: str,
    length: int,
) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} integers")
    return tuple(
        _positive_integer(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return converted


def _strict_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _rounded_float32(value: float) -> float:
    return struct.unpack("=f", struct.pack("=f", value))[0]


def _validated_teacher_noise_scale(
    value: object,
    *,
    batch_size: int,
    minimum_scale: float,
    maximum_scale: float,
    target_dtype: str,
) -> jt.Var:
    if not isinstance(value, jt.Var):
        raise TypeError("noise_scale must be a jittor.Var")
    if value.ndim == 1:
        if int(value.shape[0]) != batch_size:
            raise ValueError(
                "noise_scale batch dimension must match noisy_points"
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
    inspected = (
        scale if str(scale.dtype) == "float32" else scale.float32()
    )
    if not bool(jt.isfinite(inspected).all().item()):
        raise ValueError("noise_scale must be finite")
    if str(scale.dtype) == "float64":
        range_values = scale
        accepted_minimum = minimum_scale
        accepted_maximum = maximum_scale
    else:
        range_values = inspected
        accepted_minimum = _rounded_float32(minimum_scale)
        accepted_maximum = _rounded_float32(maximum_scale)
    actual_minimum = float(range_values.min().item())
    actual_maximum = float(range_values.max().item())
    if actual_minimum <= 0.0:
        raise ValueError("noise_scale must be positive")
    if (
        actual_minimum < accepted_minimum
        or actual_maximum > accepted_maximum
    ):
        raise ValueError(
            "noise_scale must be within the configured conditioning range "
            f"[{minimum_scale}, {maximum_scale}]"
        )
    return scale.cast(target_dtype)


class PGDDenoiser(nn.Module):
    """One-step PGD residual denoiser following commit ``b5dc8be``.

    Inputs and outputs are batch-first point patches ``(B,N,3)``.  The
    backbone predicts a bounded displacement and this module returns
    ``noisy + displacement``.  The default dimensions, codebooks, and point
    ledger match the fixed source commit; smaller valid dimensions are
    accepted only to make numerical integration tests inexpensive.
    """

    def __init__(
        self,
        *,
        patch_size: int = 1000,
        feature_dims: Sequence[int] = DEFAULT_FEATURE_DIMS,
        codebook_sizes: Sequence[int] = DEFAULT_CODEBOOK_SIZES,
        temperature: float = 0.1,
        ema_momentum: float = 0.99,
        displacement_scale: float = 1.0,
        neighbor_count: int = 16,
        interpolation_neighbors: int = 8,
        noise_conditioning: bool = False,
        pointwise_displacement_gate: bool = False,
        residual_refinement: bool = False,
        residual_refinement_hidden_dim: int = 32,
        residual_refinement_scale: float = 0.005,
        conditioning_minimum_scale: float = 0.005,
        conditioning_maximum_scale: float = 0.020,
    ) -> None:
        super().__init__()
        self.patch_size = _positive_integer(
            patch_size, name="patch_size"
        )
        self.feature_dims = _integer_tuple(
            feature_dims, name="feature_dims", length=5
        )
        self.codebook_sizes = _integer_tuple(
            codebook_sizes, name="codebook_sizes", length=4
        )
        self.neighbor_count = _positive_integer(
            neighbor_count, name="neighbor_count"
        )
        if self.neighbor_count != 16:
            raise ValueError("the code-faithful PGD uses 16 local neighbors")
        self.interpolation_neighbors = _positive_integer(
            interpolation_neighbors,
            name="interpolation_neighbors",
        )
        if self.interpolation_neighbors != 8:
            raise ValueError(
                "the code-faithful PGD uses 8 interpolation neighbors"
            )
        self.displacement_scale = _positive_finite(
            displacement_scale, name="displacement_scale"
        )
        self.noise_conditioning = _strict_bool(
            noise_conditioning,
            name="noise_conditioning",
        )
        self.pointwise_displacement_gate = _strict_bool(
            pointwise_displacement_gate,
            name="pointwise_displacement_gate",
        )
        self.residual_refinement = _strict_bool(
            residual_refinement,
            name="residual_refinement",
        )
        self.residual_refinement_hidden_dim = _positive_integer(
            residual_refinement_hidden_dim,
            name="residual_refinement_hidden_dim",
        )
        self.residual_refinement_scale = _positive_finite(
            residual_refinement_scale,
            name="residual_refinement_scale",
        )
        self.conditioning_minimum_scale = _positive_finite(
            conditioning_minimum_scale,
            name="conditioning_minimum_scale",
        )
        self.conditioning_maximum_scale = _positive_finite(
            conditioning_maximum_scale,
            name="conditioning_maximum_scale",
        )
        if (
            self.conditioning_maximum_scale
            <= self.conditioning_minimum_scale
        ):
            raise ValueError(
                "conditioning_maximum_scale must be greater than "
                "conditioning_minimum_scale"
            )

        ledger = official_pgd_shape_ledger(self.patch_size)
        if any(dimension % 2 for dimension in self.feature_dims[1:]):
            raise ValueError(
                "all downsampling output feature dimensions must be even"
            )
        for index, stage in enumerate(ledger.decoder):
            sparse_channels = self.feature_dims[4 - index]
            validate_decoder_reshape(
                dense_points=stage.dense_points,
                sparse_points=stage.sparse_points,
                attention_channels=sparse_channels,
                k_sample=stage.k_sample,
            )

        self.start = StartBlock(
            0,
            self.feature_dims[0],
            nsample=self.neighbor_count,
            stride=1,
        )
        downsampling = []
        for index, stride in enumerate(DEFAULT_DOWNSAMPLE_STRIDES):
            downsampling.append(
                Downsampling(
                    self.feature_dims[index],
                    self.feature_dims[index + 1],
                    nsample=self.neighbor_count,
                    stride=stride,
                    enforce_exact_ratio=True,
                )
            )
        self.downsampling = nn.ModuleList(downsampling)

        upsampling = []
        dense_dimensions = tuple(reversed(self.feature_dims[:-1]))
        sparse_dimensions = tuple(reversed(self.feature_dims[1:]))
        for index, (sparse_dimension, dense_dimension) in enumerate(
            zip(sparse_dimensions, dense_dimensions)
        ):
            upsampling.append(
                Upsampling(
                    [sparse_dimension, dense_dimension],
                    dense_dimension,
                    nsample=self.neighbor_count,
                    stride=index + 1,
                )
            )
        self.upsampling = nn.ModuleList(upsampling)
        self.codebooks = nn.ModuleList(
            [
                SoftVectorQuantizer(
                    feature_dim=feature_dimension,
                    codebook_size=codebook_size,
                    temperature=temperature,
                    momentum=ema_momentum,
                    commitment_cost=0.0,
                    use_ema=True,
                    auto_apply_pending=True,
                )
                for feature_dimension, codebook_size in zip(
                    dense_dimensions, self.codebook_sizes
                )
            ]
        )

        self.head_linear1 = nn.Linear(
            self.feature_dims[0], 128, bias=False
        )
        self.head_linear2 = nn.Linear(128, 64)
        self.head_linear3 = nn.Linear(64, 3)

        # Optional modules are deliberately constructed after every fixed-source
        # module above.  Disabled/default construction therefore preserves the
        # original parameter names, values, and random initialization order.
        if self.noise_conditioning:
            self.noise_scale_estimator = NoiseScaleEstimator(
                feature_dim=self.feature_dims[-1],
                hidden_dim=32,
                minimum_scale=self.conditioning_minimum_scale,
                maximum_scale=self.conditioning_maximum_scale,
            )
            self.noise_scale_film = ScalarFiLM(
                feature_dim=64,
                embedding_dim=16,
            )
        else:
            self.noise_scale_estimator = None
            self.noise_scale_film = None
        if self.pointwise_displacement_gate:
            self.displacement_gate = PointwiseDisplacementGate(
                feature_dim=self.feature_dims[0],
                hidden_dim=32,
            )
        else:
            self.displacement_gate = None
        if self.residual_refinement:
            self.residual_refinement_linear1 = nn.Linear(
                self.feature_dims[0],
                self.residual_refinement_hidden_dim,
            )
            self.residual_refinement_linear2 = nn.Linear(
                self.residual_refinement_hidden_dim,
                3,
            )
            self.residual_refinement_linear2.weight.update(
                jt.zeros_like(self.residual_refinement_linear2.weight)
            )
            self.residual_refinement_linear2.bias.update(
                jt.zeros_like(self.residual_refinement_linear2.bias)
            )
        else:
            self.residual_refinement_linear1 = None
            self.residual_refinement_linear2 = None

    def apply_pending_ema(self) -> int:
        """Commit every codebook update after the optimizer step."""

        return sum(
            bool(codebook.apply_pending_ema())
            for codebook in self.codebooks
        )

    def codebook_metrics(self) -> dict[str, jt.Var]:
        """Return stage-prefixed VQ utilization diagnostics."""

        metrics: dict[str, jt.Var] = {}
        for index, codebook in enumerate(self.codebooks, start=1):
            for name, value in codebook.metrics().items():
                metrics[f"up_{index}/{name}"] = value
        return metrics

    def execute(
        self,
        noisy_points: jt.Var,
        *,
        noise_scale: jt.Var | None = None,
        calculate_commitment_losses: bool | None = None,
        return_details: bool = False,
    ) -> jt.Var | tuple[jt.Var, dict[str, object]]:
        if not isinstance(return_details, bool):
            raise ValueError("return_details must be a bool")
        if calculate_commitment_losses is None:
            calculate_commitment = bool(self.is_training())
        elif isinstance(calculate_commitment_losses, bool):
            calculate_commitment = calculate_commitment_losses
        else:
            raise ValueError(
                "calculate_commitment_losses must be a bool or None"
            )

        points = _validate_batched_points(
            noisy_points,
            "noisy_points",
            require_finite=True,
        )
        if int(points.shape[1]) != self.patch_size:
            raise ValueError(
                f"noisy_points point count must equal patch_size "
                f"{self.patch_size}"
            )
        # This validates the complete official point ledger before any graph is
        # constructed.  All internal blocks can then skip redundant finite
        # reductions and use their batch-preserving operators.
        official_pgd_shape_ledger(int(points.shape[1]))
        if not self.noise_conditioning:
            if noise_scale is not None:
                raise ValueError(
                    "noise_scale cannot be supplied when noise conditioning "
                    "is disabled"
                )
            teacher_noise_scale = None
        elif noise_scale is None:
            teacher_noise_scale = None
        else:
            teacher_noise_scale = _validated_teacher_noise_scale(
                noise_scale,
                batch_size=int(points.shape[0]),
                minimum_scale=self.conditioning_minimum_scale,
                maximum_scale=self.conditioning_maximum_scale,
                target_dtype=str(points.dtype),
            )

        encoder_states: list[tuple[jt.Var, jt.Var]] = []
        sampling_indices: list[jt.Var] = []
        current_points, current_features = self.start(
            points,
            None,
            validate_finite=False,
        )
        encoder_states.append((current_points, current_features))
        for block in self.downsampling:
            (
                current_points,
                current_features,
                stage_indices,
            ) = block(
                current_points,
                current_features,
                validate_finite=False,
            )
            encoder_states.append((current_points, current_features))
            sampling_indices.append(stage_indices)

        estimated_noise_scale = None
        conditioned_noise_scale = None
        conditioning_source = None
        if self.noise_conditioning:
            if (
                self.noise_scale_estimator is None
                or self.noise_scale_film is None
            ):
                raise RuntimeError(
                    "noise conditioning modules were not constructed"
                )
            estimated_noise_scale = self.noise_scale_estimator(
                current_features,
                validate_finite=False,
            )
            if teacher_noise_scale is None:
                conditioned_noise_scale = estimated_noise_scale
                conditioning_source = "estimated"
            else:
                conditioned_noise_scale = teacher_noise_scale
                conditioning_source = "teacher"

        encoder_point_counts = tuple(
            int(coordinates.shape[1])
            for coordinates, _ in encoder_states
        )
        encoder_channels = tuple(
            int(features.shape[2]) for _, features in encoder_states
        )
        decoder_point_counts: list[int] = []
        decoder_channels: list[int] = []
        total_commitment = jt.zeros(
            (1,), dtype=points.dtype
        ).sum().stop_grad()

        current_points, current_features = encoder_states[-1]
        for index, (block, codebook) in enumerate(
            zip(self.upsampling, self.codebooks)
        ):
            dense_points, dense_features = encoder_states[-2 - index]
            (
                current_points,
                current_features,
                commitment,
            ) = block(
                dense_points,
                dense_features,
                current_points,
                current_features,
                sampling_indices=sampling_indices[-1 - index],
                codebook=codebook,
                calculate_commitment_loss_for_block=calculate_commitment,
                validate_finite=False,
            )
            total_commitment = total_commitment + commitment
            decoder_point_counts.append(int(current_points.shape[1]))
            decoder_channels.append(int(current_features.shape[2]))

        hidden = nn.relu(self.head_linear1(current_features))
        hidden = nn.relu(self.head_linear2(hidden))
        if self.noise_conditioning:
            if (
                self.noise_scale_film is None
                or conditioned_noise_scale is None
            ):
                raise RuntimeError(
                    "conditioned noise scale was not materialized"
                )
            hidden = self.noise_scale_film(
                hidden,
                conditioned_noise_scale,
                validate_finite=False,
            )
        primary_displacement = (
            jt.tanh(self.head_linear3(hidden))
            * self.displacement_scale
        )
        displacement_gate = None
        if self.pointwise_displacement_gate:
            if self.displacement_gate is None:
                raise RuntimeError(
                    "pointwise displacement gate was not constructed"
                )
            displacement_gate = self.displacement_gate(
                current_features,
                validate_finite=False,
            )
            primary_displacement = (
                primary_displacement * displacement_gate
            )
        refinement_displacement = None
        displacement = primary_displacement
        if self.residual_refinement:
            if (
                self.residual_refinement_linear1 is None
                or self.residual_refinement_linear2 is None
            ):
                raise RuntimeError(
                    "residual refinement modules were not constructed"
                )
            refinement_hidden = nn.relu(
                self.residual_refinement_linear1(current_features)
            )
            refinement_displacement = (
                jt.tanh(
                    self.residual_refinement_linear2(refinement_hidden)
                )
                * self.residual_refinement_scale
            )
            displacement = primary_displacement + refinement_displacement
        denoised = points + displacement
        if not return_details:
            return denoised
        details: dict[str, object] = {
            "displacement": displacement,
            "commitment_loss": total_commitment,
            "encoder_point_counts": encoder_point_counts,
            "encoder_channels": encoder_channels,
            "decoder_point_counts": tuple(decoder_point_counts),
            "decoder_channels": tuple(decoder_channels),
            "vq_forward_count": len(self.codebooks),
        }
        if self.noise_conditioning:
            details.update(
                {
                    "estimated_noise_scale": estimated_noise_scale,
                    "conditioned_noise_scale": conditioned_noise_scale,
                    "conditioning_source": conditioning_source,
                }
            )
        if self.pointwise_displacement_gate:
            details["displacement_gate"] = displacement_gate
        if self.residual_refinement:
            details.update(
                {
                    "primary_displacement": primary_displacement,
                    "refinement_displacement": refinement_displacement,
                }
            )
        return denoised, details


__all__ = [
    "DEFAULT_CODEBOOK_SIZES",
    "DEFAULT_DOWNSAMPLE_STRIDES",
    "DEFAULT_FEATURE_DIMS",
    "PGDDenoiser",
]

"""Batch-first pure-Jittor blocks from the fixed PGD source profile."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import jittor as jt
from jittor import nn

from pcdenoise.ops.fps import farthest_point_sample
from pcdenoise.ops.indexing import (
    _index_points_unchecked,
    _validate_batched_points,
)
from pcdenoise.ops.interpolate import inverse_distance_interpolate
from pcdenoise.ops.knn import knn


_FLOAT_DTYPES = {"float16", "float32", "float64", "bfloat16"}


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _validate_features(
    value: object,
    name: str,
    *,
    channels: int,
    batch_size: int | None = None,
    point_count: int | None = None,
    require_finite: bool = True,
) -> jt.Var:
    features = _validate_batched_points(
        value,
        name,
        require_xyz=False,
        require_finite=require_finite,
    )
    if int(features.shape[2]) != channels:
        raise ValueError(f"{name} must have {channels} channels")
    if batch_size is not None and int(features.shape[0]) != batch_size:
        raise ValueError(f"{name} batch dimension does not match")
    if point_count is not None and int(features.shape[1]) != point_count:
        raise ValueError(f"{name} point dimension does not match")
    return features


def _validate_sampling_indices(
    indices: object | None,
    *,
    batch_size: int,
) -> jt.Var | None:
    if indices is None:
        return None
    if not isinstance(indices, jt.Var):
        raise TypeError("sampling_indices must be a Jittor Var or None")
    if indices.ndim < 2 or int(indices.shape[0]) != batch_size:
        raise ValueError(
            "sampling_indices must have shape (B, ...) with matching batch"
        )
    if str(indices.dtype) not in {"int32", "int64"}:
        raise ValueError("sampling_indices must have integer dtype")
    return indices


def _flat_linear_norm(
    values: jt.Var,
    linear: nn.Linear,
    norm: nn.BatchNorm1d,
) -> jt.Var:
    shape = tuple(int(size) for size in values.shape)
    output = linear(values.reshape((-1, shape[-1])))
    output = norm(output)
    return output.reshape(shape[:-1] + (int(output.shape[-1]),))


@dataclass(frozen=True)
class EncoderStageShape:
    name: str
    points: int
    channels: int


@dataclass(frozen=True)
class DecoderStageShape:
    name: str
    k_sample: int
    dense_points: int
    sparse_points: int
    attention_channels: int
    projected_channels: int
    output_channels: int


@dataclass(frozen=True)
class PGDShapeLedger:
    encoder: tuple[EncoderStageShape, ...]
    decoder: tuple[DecoderStageShape, ...]


def validate_decoder_reshape(
    *,
    dense_points: int,
    sparse_points: int,
    attention_channels: int,
    k_sample: int,
) -> int:
    """Validate the fixed source's sparse-channel to dense-point reshape."""

    dense = _positive_integer(dense_points, name="dense_points")
    sparse = _positive_integer(sparse_points, name="sparse_points")
    channels = _positive_integer(
        attention_channels, name="attention_channels"
    )
    stride = _positive_integer(k_sample, name="k_sample")
    if dense <= 16:
        raise ValueError(
            "official local attention requires dense_points greater than 16"
        )
    projected = channels * (stride + 1) // stride
    if sparse * projected != dense * channels:
        raise ValueError(
            "decoder reshape requires "
            "sparse_points * projected_channels == "
            "dense_points * attention_channels"
        )
    return projected


def official_pgd_shape_ledger(point_count: int = 1000) -> PGDShapeLedger:
    """Return and validate the production PGD point/channel ledger."""

    points = _positive_integer(point_count, name="point_count")
    if points < 45 or points % 5 != 0:
        raise ValueError(
            "official PGD point_count must be a multiple of 5 and at least 45"
        )
    counts = [points]
    for stride in (4, 3, 2, 1):
        numerator = counts[-1] * stride
        denominator = stride + 1
        if numerator % denominator:
            raise ValueError("official PGD downsample ratios must be exact")
        counts.append(numerator // denominator)

    encoder_channels = (32, 48, 72, 108, 162)
    encoder_names = ("start", "down_1", "down_2", "down_3", "down_4")
    encoder = tuple(
        EncoderStageShape(name, count, channels)
        for name, count, channels in zip(
            encoder_names, counts, encoder_channels
        )
    )

    dense_counts = tuple(reversed(counts[:-1]))
    sparse_counts = tuple(reversed(counts[1:]))
    attention_channels = (162, 108, 72, 48)
    output_channels = (108, 72, 48, 32)
    strides = (1, 2, 3, 4)
    decoder = []
    for index, (dense, sparse, channels, output, stride) in enumerate(
        zip(
            dense_counts,
            sparse_counts,
            attention_channels,
            output_channels,
            strides,
        ),
        start=1,
    ):
        projected = validate_decoder_reshape(
            dense_points=dense,
            sparse_points=sparse,
            attention_channels=channels,
            k_sample=stride,
        )
        decoder.append(
            DecoderStageShape(
                name=f"up_{index}",
                k_sample=stride,
                dense_points=dense,
                sparse_points=sparse,
                attention_channels=channels,
                projected_channels=projected,
                output_channels=output,
            )
        )
    return PGDShapeLedger(encoder=encoder, decoder=tuple(decoder))


class StartBlock(nn.Module):
    """Initial coordinate/input-feature projection."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        nsample: int = 16,
        stride: int = 1,
    ) -> None:
        super().__init__()
        if isinstance(d_in, bool) or not isinstance(d_in, Integral) or d_in < 0:
            raise ValueError("d_in must be a nonnegative integer")
        self.d_in = int(d_in)
        self.d_out = _positive_integer(d_out, name="d_out")
        self.nsample = _positive_integer(nsample, name="nsample")
        if self.nsample != 16:
            raise ValueError("the code-faithful StartBlock uses nsample=16")
        self.stride = _positive_integer(stride, name="stride")
        self.linear = nn.Linear(self.d_in + 3, self.d_out)
        self.norm = nn.BatchNorm1d(self.d_out)

    def execute(
        self,
        coordinates: jt.Var,
        features: jt.Var | None = None,
        *,
        validate_finite: bool = True,
    ) -> tuple[jt.Var, jt.Var]:
        check_finite = _validate_bool(
            validate_finite, name="validate_finite"
        )
        points = _validate_batched_points(
            coordinates,
            "coordinates",
            require_finite=check_finite,
        )
        if self.d_in == 0:
            if features is not None:
                raise ValueError(
                    "features must be None when StartBlock d_in is zero"
                )
            inputs = points
        else:
            input_features = _validate_features(
                features,
                "features",
                channels=self.d_in,
                batch_size=int(points.shape[0]),
                point_count=int(points.shape[1]),
                require_finite=check_finite,
            )
            inputs = jt.concat((points, input_features), dim=-1)
        output = _flat_linear_norm(inputs, self.linear, self.norm)
        return points, nn.leaky_relu(output, scale=0.2)


class RFE(nn.Module):
    """Relation feature encoding over exactly sixteen local neighbors."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        neighbor_count: int = 16,
    ) -> None:
        super().__init__()
        self.d_in = _positive_integer(d_in, name="d_in")
        self.d_out = _positive_integer(d_out, name="d_out")
        if self.d_in != self.d_out:
            raise ValueError("the fixed-source RFE requires d_in == d_out")
        self.neighbor_count = _positive_integer(
            neighbor_count, name="neighbor_count"
        )
        if self.neighbor_count != 16:
            raise ValueError("the code-faithful RFE uses 16 neighbors")

        self.position_conv1 = nn.Conv1d(10, 2 * self.d_out, 1)
        self.position_norm = nn.BatchNorm1d(2 * self.d_out)
        self.position_conv2 = nn.Conv1d(2 * self.d_out, self.d_out, 1)
        self.score_linear = nn.Linear(
            2 * self.d_out, 2 * self.d_out, bias=False
        )
        self.output_linear = nn.Linear(2 * self.d_out, self.d_out)
        self.output_norm = nn.BatchNorm1d(self.d_out)

    def _validate_local_coordinates(
        self,
        coordinates: object,
        *,
        require_finite: bool,
    ) -> jt.Var:
        if not isinstance(coordinates, jt.Var):
            raise TypeError("local_coordinates must be a Jittor Var")
        if (
            coordinates.ndim != 3
            or int(coordinates.shape[0]) <= 0
            or int(coordinates.shape[1]) != self.neighbor_count
            or int(coordinates.shape[2]) != 3
        ):
            raise ValueError(
                "local_coordinates must have shape (M, 16, 3), M > 0"
            )
        if str(coordinates.dtype) not in _FLOAT_DTYPES:
            raise ValueError("local_coordinates must have floating dtype")
        if require_finite and not bool(
            jt.isfinite(coordinates.float32()).all().item()
        ):
            raise ValueError("local_coordinates must contain only finite values")
        return coordinates

    def _validate_local_features(
        self,
        features: object,
        *,
        example_count: int,
        require_finite: bool,
    ) -> jt.Var:
        if not isinstance(features, jt.Var):
            raise TypeError("local_features must be a Jittor Var")
        if (
            features.ndim != 3
            or int(features.shape[0]) != example_count
            or int(features.shape[1]) != self.neighbor_count
            or int(features.shape[2]) != self.d_in
        ):
            raise ValueError(
                "local_features must have shape (M, 16, d_in)"
            )
        if str(features.dtype) not in _FLOAT_DTYPES:
            raise ValueError("local_features must have floating dtype")
        if require_finite and not bool(
            jt.isfinite(features.float32()).all().item()
        ):
            raise ValueError("local_features must contain only finite values")
        return features

    def geometry_features(
        self,
        local_coordinates: jt.Var,
        *,
        validate_finite: bool = True,
    ) -> jt.Var:
        check_finite = _validate_bool(
            validate_finite, name="validate_finite"
        )
        coordinates = self._validate_local_coordinates(
            local_coordinates, require_finite=check_finite
        )
        center = coordinates[:, :1, :].broadcast(coordinates.shape)
        delta = center - coordinates
        distance = jt.sqrt(
            jt.maximum(
                (delta * delta).sum(dim=-1, keepdims=True),
                jt.zeros(
                    (
                        int(coordinates.shape[0]),
                        self.neighbor_count,
                        1,
                    ),
                    dtype=coordinates.dtype,
                ),
            )
        )
        return jt.concat(
            (center, coordinates, delta, distance), dim=-1
        )

    def execute(
        self,
        local_coordinates: jt.Var,
        local_features: jt.Var,
        *,
        validate_finite: bool = True,
    ) -> jt.Var:
        check_finite = _validate_bool(
            validate_finite, name="validate_finite"
        )
        coordinates = self._validate_local_coordinates(
            local_coordinates, require_finite=check_finite
        )
        features = self._validate_local_features(
            local_features,
            example_count=int(coordinates.shape[0]),
            require_finite=check_finite,
        )
        geometry = self.geometry_features(
            coordinates, validate_finite=False
        )
        positional = geometry.transpose(0, 2, 1)
        positional = self.position_conv1(positional)
        positional = nn.relu(self.position_norm(positional))
        positional = self.position_conv2(positional).transpose(0, 2, 1)

        fused = jt.concat((positional, features), dim=-1)
        scores = nn.softmax(self.score_linear(fused), dim=1)
        pooled = (scores * fused).sum(dim=1, keepdims=True).squeeze(1)
        output = self.output_norm(self.output_linear(pooled))
        return nn.relu(output)


class MRE(nn.Module):
    """Two-stage multi-relation encoding on a batch-local self 16-NN graph."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        neighbor_count: int = 16,
    ) -> None:
        super().__init__()
        self.d_in = _positive_integer(d_in, name="d_in")
        self.d_out = _positive_integer(d_out, name="d_out")
        if self.d_out % 2:
            raise ValueError("MRE d_out must be even")
        self.neighbor_count = _positive_integer(
            neighbor_count, name="neighbor_count"
        )
        if self.neighbor_count != 16:
            raise ValueError("the code-faithful MRE uses 16 neighbors")
        hidden = self.d_out // 2
        self.input_linear = nn.Linear(self.d_in, hidden)
        self.input_norm = nn.BatchNorm1d(hidden)
        self.fusion_linear = nn.Linear(self.d_out, self.d_out)
        self.fusion_norm = nn.BatchNorm1d(self.d_out)
        self.shortcut_linear = nn.Linear(self.d_in, self.d_out)
        self.shortcut_norm = nn.BatchNorm1d(self.d_out)
        self.rfe1 = RFE(hidden, hidden, neighbor_count=self.neighbor_count)
        self.rfe2 = RFE(hidden, hidden, neighbor_count=self.neighbor_count)

    def group_neighbors(
        self,
        coordinates: jt.Var,
        features: jt.Var,
        *,
        validate_finite: bool = True,
    ) -> tuple[jt.Var, jt.Var, jt.Var]:
        check_finite = _validate_bool(
            validate_finite, name="validate_finite"
        )
        points = _validate_batched_points(
            coordinates,
            "coordinates",
            require_finite=check_finite,
        )
        values = _validate_batched_points(
            features,
            "features",
            require_xyz=False,
            require_finite=check_finite,
        )
        if (
            int(values.shape[0]) != int(points.shape[0])
            or int(values.shape[1]) != int(points.shape[1])
        ):
            raise ValueError(
                "features must match coordinate batch and point dimensions"
            )
        if int(points.shape[1]) < self.neighbor_count:
            raise ValueError("MRE requires at least 16 points")
        _, indices = knn(
            points,
            points,
            self.neighbor_count,
            validate_finite=False,
        )
        selected_points = _index_points_unchecked(points, indices)
        selected_features = _index_points_unchecked(values, indices)
        relative = selected_points - points.unsqueeze(2)
        return relative, selected_features, indices

    def execute(
        self,
        coordinates: jt.Var,
        features: jt.Var,
        *,
        validate_finite: bool = True,
    ) -> jt.Var:
        check_finite = _validate_bool(
            validate_finite, name="validate_finite"
        )
        points = _validate_batched_points(
            coordinates,
            "coordinates",
            require_finite=check_finite,
        )
        values = _validate_features(
            features,
            "features",
            channels=self.d_in,
            batch_size=int(points.shape[0]),
            point_count=int(points.shape[1]),
            require_finite=check_finite,
        )
        if int(points.shape[1]) < self.neighbor_count:
            raise ValueError("MRE requires at least 16 points")
        batch_size, point_count = int(points.shape[0]), int(points.shape[1])
        shortcut_input = values

        projected = _flat_linear_norm(
            values, self.input_linear, self.input_norm
        )
        projected = nn.relu(projected)
        relative, grouped, _ = self.group_neighbors(
            points, projected, validate_finite=False
        )
        first = self.rfe1(
            relative.reshape((-1, self.neighbor_count, 3)),
            grouped.reshape(
                (-1, self.neighbor_count, self.d_out // 2)
            ),
            validate_finite=False,
        ).reshape((batch_size, point_count, self.d_out // 2))
        middle = first

        relative, grouped, _ = self.group_neighbors(
            points, first, validate_finite=False
        )
        second = self.rfe2(
            relative.reshape((-1, self.neighbor_count, 3)),
            grouped.reshape(
                (-1, self.neighbor_count, self.d_out // 2)
            ),
            validate_finite=False,
        ).reshape((batch_size, point_count, self.d_out // 2))

        fused = jt.concat((middle, second), dim=-1)
        fused = _flat_linear_norm(
            fused, self.fusion_linear, self.fusion_norm
        )
        fused = nn.relu(fused)
        shortcut = _flat_linear_norm(
            shortcut_input, self.shortcut_linear, self.shortcut_norm
        )
        shortcut = nn.relu(shortcut)
        return shortcut + fused


class Downsampling(nn.Module):
    """MRE followed by deterministic batch-local farthest-point sampling."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        nsample: int,
        stride: int,
        *,
        enforce_exact_ratio: bool = True,
    ) -> None:
        super().__init__()
        self.d_in = _positive_integer(d_in, name="d_in")
        self.d_out = _positive_integer(d_out, name="d_out")
        self.nsample = _positive_integer(nsample, name="nsample")
        if self.nsample != 16:
            raise ValueError("the code-faithful Downsampling uses nsample=16")
        self.stride = _positive_integer(stride, name="stride")
        self.enforce_exact_ratio = _validate_bool(
            enforce_exact_ratio, name="enforce_exact_ratio"
        )
        self.mre = MRE(
            self.d_in, self.d_out, neighbor_count=self.nsample
        )

    def execute(
        self,
        coordinates: jt.Var,
        features: jt.Var,
        *,
        validate_finite: bool = True,
    ) -> tuple[jt.Var, jt.Var, jt.Var]:
        check_finite = _validate_bool(
            validate_finite, name="validate_finite"
        )
        points = _validate_batched_points(
            coordinates,
            "coordinates",
            require_finite=check_finite,
        )
        values = _validate_features(
            features,
            "features",
            channels=self.d_in,
            batch_size=int(points.shape[0]),
            point_count=int(points.shape[1]),
            require_finite=check_finite,
        )
        point_count = int(points.shape[1])
        if point_count < self.nsample:
            raise ValueError("Downsampling MRE requires at least 16 points")
        numerator = point_count * self.stride
        denominator = self.stride + 1
        if self.enforce_exact_ratio and numerator % denominator:
            raise ValueError("downsample point ratio must be exact")
        sample_count = numerator // denominator
        aggregated = self.mre(
            points, values, validate_finite=False
        )
        indices = farthest_point_sample(
            points,
            sample_count,
            validate_finite=False,
        )
        return (
            _index_points_unchecked(points, indices),
            _index_points_unchecked(aggregated, indices),
            indices,
        )


class FiLMLayer(nn.Module):
    """Generate per-channel gamma/beta for Key-only modulation."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = _positive_integer(input_dim, name="input_dim")
        self.output_dim = _positive_integer(output_dim, name="output_dim")
        self.linear1 = nn.Linear(self.input_dim, 2 * self.input_dim)
        self.linear2 = nn.Linear(2 * self.input_dim, 2 * self.output_dim)

    def execute(
        self,
        features: jt.Var,
        *,
        validate_finite: bool = True,
    ) -> tuple[jt.Var, jt.Var]:
        check_finite = _validate_bool(
            validate_finite, name="validate_finite"
        )
        values = _validate_features(
            features,
            "features",
            channels=self.input_dim,
            require_finite=check_finite,
        )
        hidden = nn.leaky_relu(self.linear1(values), scale=0.2)
        parameters = self.linear2(hidden)
        gamma, beta = parameters.chunk(2, dim=-1)
        return gamma, beta


class CrossAttentionPointTransformerLayer(nn.Module):
    """Fixed-source local vector attention with sparse K/V reshaping."""

    def __init__(
        self,
        dim: int,
        dim_dense: int,
        k_sample: int,
        attn_mlp_hidden_mult: int = 4,
        num_neighbors: int = 16,
    ) -> None:
        super().__init__()
        self.dim = _positive_integer(dim, name="dim")
        self.dim_dense = _positive_integer(
            dim_dense, name="dim_dense"
        )
        self.k_sample = _positive_integer(k_sample, name="k_sample")
        self.attn_mlp_hidden_mult = _positive_integer(
            attn_mlp_hidden_mult, name="attn_mlp_hidden_mult"
        )
        self.num_neighbors = _positive_integer(
            num_neighbors, name="num_neighbors"
        )
        if self.num_neighbors != 16:
            raise ValueError(
                "the code-faithful cross attention uses 16 neighbors"
            )
        self.projected_dim = (
            self.dim * (self.k_sample + 1) // self.k_sample
        )
        self.to_q = nn.Linear(self.dim, self.dim, bias=False)
        self.to_k = nn.Linear(
            self.dim, self.projected_dim, bias=False
        )
        self.to_v = nn.Linear(
            self.dim, self.projected_dim, bias=False
        )
        self.film_layer = FiLMLayer(self.dim_dense, self.dim)
        hidden = self.dim * self.attn_mlp_hidden_mult
        self.attention_linear1 = nn.Linear(self.dim, hidden)
        self.attention_norm = nn.BatchNorm1d(hidden)
        self.attention_linear2 = nn.Linear(hidden, self.dim)

    def _validated_qkv_inputs(
        self,
        query_features: jt.Var,
        sparse_keys: jt.Var,
        sparse_values: jt.Var,
        *,
        require_finite: bool,
    ) -> tuple[jt.Var, jt.Var, jt.Var, int, int, int]:
        query = _validate_features(
            query_features,
            "query_features",
            channels=self.dim,
            require_finite=require_finite,
        )
        batch_size, dense_points = (
            int(query.shape[0]),
            int(query.shape[1]),
        )
        keys = _validate_features(
            sparse_keys,
            "sparse_keys",
            channels=self.dim,
            batch_size=batch_size,
            require_finite=require_finite,
        )
        sparse_points = int(keys.shape[1])
        values = _validate_features(
            sparse_values,
            "sparse_values",
            channels=self.dim,
            batch_size=batch_size,
            point_count=sparse_points,
            require_finite=require_finite,
        )
        validate_decoder_reshape(
            dense_points=dense_points,
            sparse_points=sparse_points,
            attention_channels=self.dim,
            k_sample=self.k_sample,
        )
        return (
            query,
            keys,
            values,
            batch_size,
            dense_points,
            sparse_points,
        )

    def project_qkv(
        self,
        query_features: jt.Var,
        sparse_keys: jt.Var,
        sparse_values: jt.Var,
        *,
        quantized_features: jt.Var | None = None,
        validate_finite: bool = True,
    ) -> tuple[jt.Var, jt.Var, jt.Var]:
        check_finite = _validate_bool(
            validate_finite, name="validate_finite"
        )
        (
            query,
            keys,
            values,
            batch_size,
            dense_points,
            _,
        ) = self._validated_qkv_inputs(
            query_features,
            sparse_keys,
            sparse_values,
            require_finite=check_finite,
        )
        q = self.to_q(query)
        k = self.to_k(keys).reshape(
            (batch_size, dense_points, self.dim)
        )
        v = self.to_v(values).reshape(
            (batch_size, dense_points, self.dim)
        )
        if quantized_features is not None:
            quantized = _validate_features(
                quantized_features,
                "quantized_features",
                channels=self.dim_dense,
                batch_size=batch_size,
                point_count=dense_points,
                require_finite=check_finite,
            )
            gamma, beta = self.film_layer(
                quantized, validate_finite=False
            )
            k = gamma * k + beta
        return q, k, v

    def execute(
        self,
        query_features: jt.Var,
        sparse_keys: jt.Var,
        sparse_values: jt.Var,
        dense_positions: jt.Var,
        sampling_indices: jt.Var | None = None,
        quantized_features: jt.Var | None = None,
        *,
        validate_finite: bool = True,
        return_details: bool = False,
    ) -> jt.Var | tuple[jt.Var, dict[str, jt.Var]]:
        check_finite = _validate_bool(
            validate_finite, name="validate_finite"
        )
        details_requested = _validate_bool(
            return_details, name="return_details"
        )
        (
            query,
            keys,
            values,
            batch_size,
            dense_points,
            _,
        ) = self._validated_qkv_inputs(
            query_features,
            sparse_keys,
            sparse_values,
            require_finite=check_finite,
        )
        positions = _validate_batched_points(
            dense_positions,
            "dense_positions",
            require_finite=check_finite,
        )
        if (
            int(positions.shape[0]) != batch_size
            or int(positions.shape[1]) != dense_points
        ):
            raise ValueError(
                "dense_positions must match query batch and point dimensions"
            )
        _validate_sampling_indices(
            sampling_indices, batch_size=batch_size
        )
        q, k, v = self.project_qkv(
            query,
            keys,
            values,
            quantized_features=quantized_features,
            validate_finite=False,
        )
        _, indices = knn(
            positions,
            positions,
            self.num_neighbors,
            validate_finite=False,
        )
        selected_k = _index_points_unchecked(k, indices)
        selected_v = _index_points_unchecked(v, indices)
        selected_dense = _index_points_unchecked(query, indices)
        relation = q.unsqueeze(2) - selected_k + selected_dense

        flattened = relation.reshape((-1, self.dim))
        similarity = self.attention_linear1(flattened)
        similarity = nn.relu(self.attention_norm(similarity))
        similarity = self.attention_linear2(similarity).reshape(
            (
                batch_size,
                dense_points,
                self.num_neighbors,
                self.dim,
            )
        )
        attention = nn.softmax(similarity, dim=2)
        output = (
            attention * (selected_v + selected_dense)
        ).sum(dim=2)
        if details_requested:
            return output, {
                "indices": indices,
                "attention": attention,
                "query": q,
                "keys": k,
                "values": v,
            }
        return output


class Upsampling(nn.Module):
    """8-NN interpolation, VQ prior injection, and local vector attention."""

    def __init__(
        self,
        d_in_sparse_fusion: list[int] | tuple[int, int],
        d_out: int,
        nsample: int,
        stride: int,
        idx_module: object | None = None,
    ) -> None:
        super().__init__()
        if (
            not isinstance(d_in_sparse_fusion, (list, tuple))
            or len(d_in_sparse_fusion) != 2
        ):
            raise ValueError(
                "d_in_sparse_fusion must be [sparse_channels, dense_channels]"
            )
        self.d_in_sparse = _positive_integer(
            d_in_sparse_fusion[0], name="d_in_sparse"
        )
        self.d_in_dense = _positive_integer(
            d_in_sparse_fusion[1], name="d_in_dense"
        )
        self.d_out = _positive_integer(d_out, name="d_out")
        self.nsample = _positive_integer(nsample, name="nsample")
        if self.nsample != 16:
            raise ValueError("the code-faithful Upsampling uses nsample=16")
        self.stride = _positive_integer(stride, name="stride")
        self.idx_module = idx_module
        self.query_linear = nn.Linear(
            self.d_in_sparse + self.d_in_dense,
            self.d_in_sparse,
        )
        self.cross_attention = CrossAttentionPointTransformerLayer(
            dim=self.d_in_sparse,
            dim_dense=self.d_in_dense,
            k_sample=self.stride,
            attn_mlp_hidden_mult=1,
            num_neighbors=self.nsample,
        )
        self.output_linear = nn.Linear(
            self.d_in_sparse + self.d_in_dense, self.d_out
        )
        self.output_norm = nn.BatchNorm1d(self.d_out)

    def execute(
        self,
        dense_points: jt.Var,
        dense_features: jt.Var,
        sparse_points: jt.Var,
        sparse_features: jt.Var,
        sampling_indices: jt.Var | None = None,
        codebook: object | None = None,
        calculate_commitment_loss_for_block: bool = False,
        *,
        validate_finite: bool = True,
    ) -> tuple[jt.Var, jt.Var, jt.Var]:
        check_finite = _validate_bool(
            validate_finite, name="validate_finite"
        )
        calculate_commitment = _validate_bool(
            calculate_commitment_loss_for_block,
            name="calculate_commitment_loss_for_block",
        )
        dense_coordinates = _validate_batched_points(
            dense_points,
            "dense_points",
            require_finite=check_finite,
        )
        batch_size, dense_count = (
            int(dense_coordinates.shape[0]),
            int(dense_coordinates.shape[1]),
        )
        dense_values = _validate_features(
            dense_features,
            "dense_features",
            channels=self.d_in_dense,
            batch_size=batch_size,
            point_count=dense_count,
            require_finite=check_finite,
        )
        sparse_coordinates = _validate_batched_points(
            sparse_points,
            "sparse_points",
            require_finite=check_finite,
        )
        if int(sparse_coordinates.shape[0]) != batch_size:
            raise ValueError("sparse_points batch dimension does not match")
        sparse_count = int(sparse_coordinates.shape[1])
        if sparse_count < 8:
            raise ValueError("Upsampling interpolation requires at least 8 points")
        sparse_values = _validate_features(
            sparse_features,
            "sparse_features",
            channels=self.d_in_sparse,
            batch_size=batch_size,
            point_count=sparse_count,
            require_finite=check_finite,
        )
        validate_decoder_reshape(
            dense_points=dense_count,
            sparse_points=sparse_count,
            attention_channels=self.d_in_sparse,
            k_sample=self.stride,
        )
        _validate_sampling_indices(
            sampling_indices, batch_size=batch_size
        )

        interpolated = inverse_distance_interpolate(
            dense_coordinates,
            sparse_coordinates,
            sparse_values,
            k=8,
            validate_finite=False,
        )
        query = self.query_linear(
            jt.concat((dense_values, interpolated), dim=-1)
        )

        if codebook is None:
            quantized = dense_values
            commitment = jt.zeros(
                (1,), dtype=dense_values.dtype
            ).sum().stop_grad()
        else:
            feature_dim = getattr(codebook, "feature_dim", None)
            if feature_dim != self.d_in_dense or not callable(codebook):
                raise ValueError(
                    "codebook must be callable with matching feature_dim"
                )
            quantized, commitment = codebook(
                dense_values,
                calculate_commitment_loss=calculate_commitment,
            )
            if tuple(quantized.shape) != tuple(dense_values.shape):
                raise ValueError(
                    "codebook output must match dense feature shape"
                )

        enhanced = self.cross_attention(
            query,
            sparse_values,
            sparse_values,
            dense_coordinates,
            sampling_indices=sampling_indices,
            quantized_features=quantized,
            validate_finite=False,
        )
        output = _flat_linear_norm(
            jt.concat((enhanced, quantized), dim=-1),
            self.output_linear,
            self.output_norm,
        )
        return dense_coordinates, nn.relu(output), commitment


__all__ = [
    "CrossAttentionPointTransformerLayer",
    "DecoderStageShape",
    "Downsampling",
    "EncoderStageShape",
    "FiLMLayer",
    "MRE",
    "PGDShapeLedger",
    "RFE",
    "StartBlock",
    "Upsampling",
    "official_pgd_shape_ledger",
    "validate_decoder_reshape",
]

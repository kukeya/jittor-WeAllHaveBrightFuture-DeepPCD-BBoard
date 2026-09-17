"""Shared, compatibility-preserving PGD model construction."""

from __future__ import annotations

from collections.abc import Mapping


MODEL_ARCHITECTURES = ("pgd",)


def pgd_model_architecture(
    model_config: Mapping[str, object],
) -> str:
    """Return the explicit architecture, defaulting legacy configs to PGD."""

    architecture = model_config.get("architecture", "pgd")
    if (
        not isinstance(architecture, str)
        or architecture not in MODEL_ARCHITECTURES
    ):
        raise ValueError(
            "model.architecture must be one of "
            f"{MODEL_ARCHITECTURES}"
        )
    return architecture


def pgd_model_kwargs(
    model_config: Mapping[str, object],
    data_config: Mapping[str, object],
) -> dict[str, object]:
    """Return PGDDenoiser kwargs shared by training and inference.

    Optional arguments are forwarded only when explicitly present in the
    canonical v2 config.  This keeps legacy constructor calls byte-for-byte
    equivalent at the argument surface while binding an enabled conditioner's
    admissible scale interval to the training data interval.
    """

    kwargs: dict[str, object] = {
        "patch_size": int(model_config["patch_size"]),
        "feature_dims": tuple(model_config["feature_dims"]),
        "codebook_sizes": tuple(model_config["codebook_sizes"]),
        "temperature": float(model_config["temperature"]),
        "ema_momentum": float(model_config["ema_momentum"]),
        "displacement_scale": float(
            model_config["displacement_scale"]
        ),
    }
    if "noise_conditioning" in model_config:
        enabled = model_config["noise_conditioning"]
        if not isinstance(enabled, bool):
            raise ValueError("model.noise_conditioning must be a bool")
        kwargs["noise_conditioning"] = enabled
        if enabled:
            has_minimum = "conditioning_noise_min" in data_config
            has_maximum = "conditioning_noise_max" in data_config
            if has_minimum != has_maximum:
                raise ValueError(
                    "data conditioning noise bounds must be supplied together"
                )
            kwargs["conditioning_minimum_scale"] = float(
                data_config[
                    "conditioning_noise_min"
                    if has_minimum
                    else "noise_min"
                ]
            )
            kwargs["conditioning_maximum_scale"] = float(
                data_config[
                    "conditioning_noise_max"
                    if has_maximum
                    else "noise_max"
                ]
            )
    if "pointwise_displacement_gate" in model_config:
        enabled = model_config["pointwise_displacement_gate"]
        if not isinstance(enabled, bool):
            raise ValueError(
                "model.pointwise_displacement_gate must be a bool"
            )
        kwargs["pointwise_displacement_gate"] = enabled
    if "residual_refinement" in model_config:
        enabled = model_config["residual_refinement"]
        if not isinstance(enabled, bool):
            raise ValueError("model.residual_refinement must be a bool")
        kwargs["residual_refinement"] = enabled
        if enabled:
            kwargs["residual_refinement_hidden_dim"] = int(
                model_config.get("residual_refinement_hidden_dim", 32)
            )
            kwargs["residual_refinement_scale"] = float(
                model_config.get("residual_refinement_scale", 0.005)
            )
    return kwargs


def build_pgd_model(
    model_config: Mapping[str, object],
    data_config: Mapping[str, object],
) -> object:
    """Construct the architecture bound by one canonical training config."""

    pgd_model_architecture(model_config)
    kwargs = pgd_model_kwargs(model_config, data_config)
    from .pgd import PGDDenoiser

    return PGDDenoiser(**kwargs)


__all__ = [
    "MODEL_ARCHITECTURES",
    "build_pgd_model",
    "pgd_model_architecture",
    "pgd_model_kwargs",
]

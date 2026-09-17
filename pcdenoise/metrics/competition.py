"""Per-sample score mapping and global competition aggregation."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


ZERO_BASELINE_THRESHOLD = 1.0e-15


def _metric(value: object, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            f"{name} must be a finite nonnegative number"
        ) from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _sample_parts(sample_id: str) -> tuple[str, str]:
    if not isinstance(sample_id, str):
        raise ValueError("sample_id must be a string")
    parts = sample_id.split("/")
    if (
        len(parts) != 2
        or not all(parts)
        or any(part in (".", "..") or "\\" in part for part in parts)
    ):
        raise ValueError("sample_id must have form <synset>/<model>")
    return parts[0], parts[1]


def metric_to_score(value_predicted: float, value_noisy: float) -> float:
    """Map one metric to [0,100] using the exact organizer edge behavior."""

    predicted = _metric(value_predicted, "value_predicted")
    noisy = _metric(value_noisy, "value_noisy")
    if noisy < ZERO_BASELINE_THRESHOLD:
        return 100.0 if predicted < ZERO_BASELINE_THRESHOLD else 0.0
    score = 100.0 * (1.0 - predicted / noisy)
    return max(0.0, min(100.0, score))


def score_sample_metrics(
    *,
    sample_id: str,
    cd_pred: float,
    cd_noisy: float,
    p2s_pred: float | None = None,
    p2s_noisy: float | None = None,
) -> dict[str, object]:
    """Score one valid prediction while retaining its raw metrics."""

    synset, _ = _sample_parts(sample_id)
    actual_cd_pred = _metric(cd_pred, "cd_pred")
    actual_cd_noisy = _metric(cd_noisy, "cd_noisy")
    cd_score = metric_to_score(actual_cd_pred, actual_cd_noisy)

    if (p2s_pred is None) != (p2s_noisy is None):
        raise ValueError("p2s_pred and p2s_noisy must be supplied together")
    if p2s_pred is None:
        actual_p2s_pred = None
        actual_p2s_noisy = None
        p2s_score = None
        total_score = cd_score
    else:
        actual_p2s_pred = _metric(p2s_pred, "p2s_pred")
        actual_p2s_noisy = _metric(p2s_noisy, "p2s_noisy")
        p2s_score = metric_to_score(
            actual_p2s_pred,
            actual_p2s_noisy,
        )
        total_score = 0.5 * cd_score + 0.5 * p2s_score

    return {
        "sample_id": sample_id,
        "synset": synset,
        "missing": False,
        "cd_pred": actual_cd_pred,
        "cd_noisy": actual_cd_noisy,
        "cd_score": cd_score,
        "p2s_pred": actual_p2s_pred,
        "p2s_noisy": actual_p2s_noisy,
        "p2s_score": p2s_score,
        "total_score": total_score,
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(sum(values) / len(values))


def aggregate_sample_scores(
    rows: Iterable[Mapping[str, object]],
    *,
    expected_sample_ids: Sequence[str] | None = None,
    require_p2s: bool,
) -> dict[str, object]:
    """Map missing samples to zero, then average globally over samples."""

    indexed: dict[str, dict[str, object]] = {}
    for source_row in rows:
        row = dict(source_row)
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str):
            raise ValueError("every row must contain a string sample_id")
        _sample_parts(sample_id)
        if sample_id in indexed:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        cd_score = _metric(row.get("cd_score"), "cd_score")
        if cd_score > 100.0:
            raise ValueError("cd_score must not exceed 100")
        row["cd_pred"] = _metric(row.get("cd_pred"), "cd_pred")
        row["cd_noisy"] = _metric(row.get("cd_noisy"), "cd_noisy")
        p2s_value = row.get("p2s_score")
        if require_p2s:
            if p2s_value is None:
                raise ValueError(f"P2S is required for sample {sample_id}")
            p2s_score = _metric(p2s_value, "p2s_score")
            if p2s_score > 100.0:
                raise ValueError("p2s_score must not exceed 100")
            row["p2s_pred"] = _metric(row.get("p2s_pred"), "p2s_pred")
            row["p2s_noisy"] = _metric(row.get("p2s_noisy"), "p2s_noisy")
        else:
            p2s_score = None
        row["cd_score"] = cd_score
        row["p2s_score"] = p2s_score
        row["total_score"] = (
            0.5 * cd_score + 0.5 * p2s_score
            if require_p2s
            else cd_score
        )
        row["missing"] = False
        indexed[sample_id] = row

    if expected_sample_ids is None:
        expected = sorted(indexed)
    else:
        expected = list(expected_sample_ids)
        for sample_id in expected:
            _sample_parts(sample_id)
        if len(set(expected)) != len(expected):
            raise ValueError("expected_sample_ids contains duplicates")
        expected.sort()
        unexpected = sorted(set(indexed) - set(expected))
        if unexpected:
            raise ValueError(f"unexpected scored samples: {unexpected[:3]}")
    if not expected:
        raise ValueError("at least one expected sample is required")

    per_sample: list[dict[str, object]] = []
    missing_ids: list[str] = []
    for sample_id in expected:
        if sample_id in indexed:
            per_sample.append(indexed[sample_id])
            continue
        synset, _ = _sample_parts(sample_id)
        missing_ids.append(sample_id)
        per_sample.append(
            {
                "sample_id": sample_id,
                "synset": synset,
                "missing": True,
                "cd_pred": None,
                "cd_noisy": None,
                "cd_score": 0.0,
                "p2s_pred": None,
                "p2s_noisy": None,
                "p2s_score": 0.0 if require_p2s else None,
                "total_score": 0.0,
            }
        )

    mean_cd = _mean([float(row["cd_score"]) for row in per_sample])
    mean_p2s = (
        _mean([float(row["p2s_score"]) for row in per_sample])
        if require_p2s
        else None
    )
    total_score = (
        0.5 * mean_cd + 0.5 * mean_p2s
        if mean_p2s is not None
        else mean_cd
    )
    valid_rows = [row for row in per_sample if not bool(row["missing"])]
    mean_cd_pred = (
        _mean([float(row["cd_pred"]) for row in valid_rows])
        if valid_rows
        else None
    )
    mean_cd_noisy = (
        _mean([float(row["cd_noisy"]) for row in valid_rows])
        if valid_rows
        else None
    )
    mean_p2s_pred = (
        _mean([float(row["p2s_pred"]) for row in valid_rows])
        if require_p2s and valid_rows
        else None
    )
    mean_p2s_noisy = (
        _mean([float(row["p2s_noisy"]) for row in valid_rows])
        if require_p2s and valid_rows
        else None
    )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in per_sample:
        grouped[str(row["synset"])].append(row)
    per_synset: dict[str, dict[str, object]] = {}
    for synset in sorted(grouped):
        group = grouped[synset]
        valid_group = [row for row in group if not bool(row["missing"])]
        category_cd = _mean([float(row["cd_score"]) for row in group])
        category_p2s = (
            _mean([float(row["p2s_score"]) for row in group])
            if require_p2s
            else None
        )
        category_total = (
            0.5 * category_cd + 0.5 * category_p2s
            if category_p2s is not None
            else category_cd
        )
        per_synset[synset] = {
            "sample_count": len(group),
            "valid_count": sum(not bool(row["missing"]) for row in group),
            "missing_count": sum(bool(row["missing"]) for row in group),
            "mean_cd_score": category_cd,
            "mean_p2s_score": category_p2s,
            "total_score": category_total,
            "mean_cd_pred": (
                _mean([float(row["cd_pred"]) for row in valid_group])
                if valid_group
                else None
            ),
            "mean_cd_noisy": (
                _mean([float(row["cd_noisy"]) for row in valid_group])
                if valid_group
                else None
            ),
            "mean_p2s_pred": (
                _mean([float(row["p2s_pred"]) for row in valid_group])
                if require_p2s and valid_group
                else None
            ),
            "mean_p2s_noisy": (
                _mean([float(row["p2s_noisy"]) for row in valid_group])
                if require_p2s and valid_group
                else None
            ),
        }

    return {
        "sample_count": len(per_sample),
        "valid_count": len(per_sample) - len(missing_ids),
        "missing_count": len(missing_ids),
        "missing_sample_ids": missing_ids,
        "mean_cd_score": mean_cd,
        "mean_p2s_score": mean_p2s,
        "total_score": total_score,
        "mean_cd_pred": mean_cd_pred,
        "mean_cd_noisy": mean_cd_noisy,
        "mean_p2s_pred": mean_p2s_pred,
        "mean_p2s_noisy": mean_p2s_noisy,
        "per_synset": per_synset,
        "per_sample": per_sample,
    }

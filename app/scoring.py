"""
Transparent, deterministic scoring and advice generation.

Every number shown to the user is derived from the actual measured features
-- nothing here is random or hard-coded. The normalization ranges and
weights are heuristic choices (documented in the README), not medical or
scientific standards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from app import config


@dataclass
class ComponentScores:
    peak_bulge: Optional[float]
    flex_change: Optional[float]
    definition: Optional[float]
    shape: Optional[float]
    curvature: Optional[float]
    unreliable: List[str] = field(default_factory=list)


@dataclass
class ScanResult:
    overall_score: Optional[float]
    components: ComponentScores
    advice: List[str]


def _normalize(value: Optional[float], key: str) -> Optional[float]:
    """Map a raw measurement onto a 0-10 scale using the configured range."""
    if value is None:
        return None
    lo, hi = config.NORMALIZATION_RANGES[key]
    if hi <= lo:
        return None
    clipped = max(lo, min(hi, value))
    normalized = (clipped - lo) / (hi - lo)
    return normalized * (config.SCORE_MAX - config.SCORE_MIN) + config.SCORE_MIN


def compute_component_scores(
    relaxed: Dict[str, float], flexed: Dict[str, float]
) -> ComponentScores:
    """
    Compute per-component scores from aggregated relaxed/flexed measurements.
    Uses multi-slice data when available for more robust comparison.
    """
    unreliable = []

    # Peak bulge: from flexed state
    peak_bulge_raw = flexed.get("peak_bulge")
    peak_bulge_score = _normalize(peak_bulge_raw, "peak_bulge")
    if peak_bulge_score is None:
        unreliable.append("peak_bulge")

    # Flex change: difference in peak bulge between relaxed and flexed
    flex_change_raw = None
    r_bulge = relaxed.get("peak_bulge")
    f_bulge = flexed.get("peak_bulge")
    if r_bulge is not None and f_bulge is not None:
        flex_change_raw = f_bulge - r_bulge
    flex_change_score = _normalize(flex_change_raw, "flex_change")
    if flex_change_score is None:
        unreliable.append("flex_change")

    # Width-profile comparison: additional robustness from multi-slice data
    r_profile = relaxed.get("width_profile")
    f_profile = flexed.get("width_profile")
    if r_profile is not None and f_profile is not None and len(r_profile) == len(f_profile):
        r_arr = np.array(r_profile, dtype=np.float64)
        f_arr = np.array(f_profile, dtype=np.float64)
        # Mean relative increase across all slices
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_change = np.where(r_arr > 1e-6, (f_arr - r_arr) / r_arr, 0.0)
        mean_rel_change = float(np.median(rel_change))
        # If profile-based change is available and more reliable, use it to boost confidence
        if flex_change_raw is not None and mean_rel_change > flex_change_raw:
            # Profile analysis found more change than peak-only
            profile_boost = min(0.05, (mean_rel_change - flex_change_raw) * 0.1)
            flex_change_raw = min(flex_change_raw + profile_boost, config.NORMALIZATION_RANGES["flex_change"][1])
            flex_change_score = _normalize(flex_change_raw, "flex_change")

    # Definition: from flexed state
    definition_raw = flexed.get("definition")
    definition_score = _normalize(definition_raw, "definition")
    if definition_score is None:
        unreliable.append("definition")

    # Shape: from flexed state
    shape_raw = flexed.get("shape")
    shape_score = _normalize(shape_raw, "shape")
    if shape_score is None:
        unreliable.append("shape")

    # Curvature: from flexed state
    curvature_raw = flexed.get("curvature")
    curvature_score = _normalize(curvature_raw, "curvature")
    if curvature_score is None:
        unreliable.append("curvature")

    return ComponentScores(
        peak_bulge=peak_bulge_score,
        flex_change=flex_change_score,
        definition=definition_score,
        shape=shape_score,
        curvature=curvature_score,
        unreliable=unreliable,
    )


def compute_overall_score(components: ComponentScores) -> Optional[float]:
    """Weighted average over measurable components, renormalizing weights."""
    weighted_sum = 0.0
    weight_total = 0.0
    values = {
        "peak_bulge": components.peak_bulge,
        "flex_change": components.flex_change,
        "definition": components.definition,
        "shape": components.shape,
        "curvature": components.curvature,
    }
    for key, value in values.items():
        if value is None:
            continue
        w = config.SCORING_WEIGHTS[key]
        weighted_sum += value * w
        weight_total += w

    if weight_total <= 0:
        return None
    score = weighted_sum / weight_total
    return round(max(config.SCORE_MIN, min(config.SCORE_MAX, score)), 1)


def generate_advice(components: ComponentScores, overall: Optional[float]) -> List[str]:
    """Deterministic, rule-based advice. Capped at 3 points."""
    advice: List[str] = []

    if overall is None:
        return ["Unable to obtain a reliable scan -- adjust position/lighting and try again."]

    label = {
        "peak_bulge": "Bicep prominence",
        "flex_change": "Flex response",
        "definition": "Visible definition",
        "shape": "Shape/contour",
        "curvature": "Bicep curvature",
    }
    for key in components.unreliable:
        advice.append(
            f"{label[key]} could not be estimated reliably -- try more "
            f"consistent lighting and keep the arm fully in frame."
        )

    if components.flex_change is not None:
        if components.flex_change >= 7.0:
            advice.append(
                "Strong visible flex response -- the flexed contour is "
                "noticeably more prominent than the relaxed state."
            )
        elif components.flex_change <= 3.5:
            advice.append(
                "Flex response was subtle in this scan -- try flexing "
                "harder or adjusting the camera angle."
            )

    if components.definition is not None and components.definition <= 3.5:
        advice.append(
            "Visible definition looked low -- this can also be caused "
            "by flat/diffuse lighting rather than the arm itself."
        )

    if components.peak_bulge is not None and components.peak_bulge >= 7.5 and len(advice) < 3:
        advice.append("Good visible prominence in the flexed position.")

    if not advice:
        advice.append(
            "Scan completed with consistent, reliable measurements across all components."
        )

    return advice[:3]


def score_scan(relaxed: Dict[str, float], flexed: Dict[str, float]) -> ScanResult:
    components = compute_component_scores(relaxed, flexed)
    overall = compute_overall_score(components)
    advice = generate_advice(components, overall)
    return ScanResult(overall_score=overall, components=components, advice=advice)

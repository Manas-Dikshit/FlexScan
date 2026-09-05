import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config
from app.scoring import (
    compute_component_scores,
    compute_overall_score,
    generate_advice,
    score_scan,
    _normalize,
)


def test_normalize_clamps_to_score_range():
    lo, hi = config.NORMALIZATION_RANGES["peak_bulge"]
    assert _normalize(lo - 10, "peak_bulge") == config.SCORE_MIN
    assert _normalize(hi + 10, "peak_bulge") == config.SCORE_MAX


def test_normalize_midpoint():
    lo, hi = config.NORMALIZATION_RANGES["definition"]
    mid = (lo + hi) / 2
    result = _normalize(mid, "definition")
    expected = (config.SCORE_MIN + config.SCORE_MAX) / 2
    assert abs(result - expected) < 1e-6


def test_normalize_none_passthrough():
    assert _normalize(None, "shape") is None


def test_compute_component_scores_all_reliable():
    relaxed = {"peak_bulge": 0.30, "definition": 0.05, "shape": 0.7, "curvature": 0.05}
    flexed = {"peak_bulge": 0.50, "definition": 0.10, "shape": 0.8, "curvature": 0.10}
    components = compute_component_scores(relaxed, flexed)
    assert components.peak_bulge is not None
    assert components.flex_change is not None
    assert components.definition is not None
    assert components.shape is not None
    assert components.curvature is not None
    assert components.unreliable == []


def test_compute_component_scores_missing_relaxed_peak_bulge():
    relaxed = {"peak_bulge": None, "definition": 0.05, "shape": 0.7, "curvature": 0.05}
    flexed = {"peak_bulge": 0.50, "definition": 0.10, "shape": 0.8, "curvature": 0.10}
    components = compute_component_scores(relaxed, flexed)
    assert components.flex_change is None
    assert "flex_change" in components.unreliable
    assert components.peak_bulge is not None


def test_flex_change_uses_median_of_real_peak_and_profile_measures():
    relaxed = {"peak_bulge": 0.30, "definition": 0.05, "shape": 0.7, "curvature": 0.05,
               "width_profile": [0.24, 0.30, 0.36], "arm_length": 150.0}
    flexed = {"peak_bulge": 0.38, "definition": 0.10, "shape": 0.8, "curvature": 0.10,
              "width_profile": [0.32, 0.38, 0.44], "arm_length": 150.5}
    components = compute_component_scores(relaxed, flexed)
    assert components.flex_change is not None
    assert abs(components.flex_change - _normalize(0.08, "flex_change")) < 1e-6


def test_flex_change_unreliable_when_phase_geometry_changed():
    relaxed = {"peak_bulge": 0.30, "definition": 0.05, "shape": 0.7, "curvature": 0.05,
               "arm_length": 100.0}
    flexed = {"peak_bulge": 0.50, "definition": 0.10, "shape": 0.8, "curvature": 0.10,
              "arm_length": 500.0}
    components = compute_component_scores(relaxed, flexed)
    assert components.flex_change is None
    assert "flex_change" in components.unreliable
    assert components.peak_bulge is not None


def test_overall_score_within_bounds():
    relaxed = {"peak_bulge": 0.30, "definition": 0.05, "shape": 0.7, "curvature": 0.05}
    flexed = {"peak_bulge": 0.55, "definition": 0.15, "shape": 0.85, "curvature": 0.12}
    components = compute_component_scores(relaxed, flexed)
    overall = compute_overall_score(components)
    assert overall is not None
    assert config.SCORE_MIN <= overall <= config.SCORE_MAX


def test_overall_score_none_when_everything_unreliable():
    relaxed = {"peak_bulge": None, "definition": None, "shape": None, "curvature": None}
    flexed = {"peak_bulge": None, "definition": None, "shape": None, "curvature": None}
    components = compute_component_scores(relaxed, flexed)
    overall = compute_overall_score(components)
    assert overall is None


def test_overall_score_renormalizes_with_partial_data():
    relaxed = {"peak_bulge": 0.30, "definition": None, "shape": None, "curvature": None}
    flexed = {"peak_bulge": 0.55, "definition": None, "shape": None, "curvature": None}
    components = compute_component_scores(relaxed, flexed)
    overall = compute_overall_score(components)
    assert overall is not None
    assert config.SCORE_MIN <= overall <= config.SCORE_MAX


def test_advice_reports_unreliable_components_without_inventing_numbers():
    relaxed = {"peak_bulge": None, "definition": None, "shape": None, "curvature": None}
    flexed = {"peak_bulge": None, "definition": None, "shape": None, "curvature": None}
    components = compute_component_scores(relaxed, flexed)
    overall = compute_overall_score(components)
    advice = generate_advice(components, overall)
    assert len(advice) >= 1
    assert overall is None


def test_advice_capped_at_three_points():
    relaxed = {"peak_bulge": 0.60, "definition": 0.01, "shape": 0.5, "curvature": 0.01}
    flexed = {"peak_bulge": 0.28, "definition": 0.01, "shape": 0.5, "curvature": 0.01}
    components = compute_component_scores(relaxed, flexed)
    overall = compute_overall_score(components)
    advice = generate_advice(components, overall)
    assert len(advice) <= 3


def test_score_scan_end_to_end():
    relaxed = {"peak_bulge": 0.32, "definition": 0.06, "shape": 0.72, "curvature": 0.05}
    flexed = {"peak_bulge": 0.58, "definition": 0.14, "shape": 0.88, "curvature": 0.12}
    result = score_scan(relaxed, flexed)
    assert result.overall_score is not None
    assert 1 <= len(result.advice) <= 3


def test_physical_dimensions_calibrated_conversion():
    # widths_cm already converted; scoring just selects/aggregates them
    relaxed = {"peak_bulge": 0.30, "definition": 0.05, "shape": 0.7, "curvature": 0.05,
               "max_width_cm": 11.2, "mean_width_cm": 10.0}
    flexed = {"peak_bulge": 0.50, "definition": 0.10, "shape": 0.8, "curvature": 0.10,
              "max_width_cm": 14.0, "mean_width_cm": 12.5}
    result = score_scan(relaxed, flexed, reference_cm=30.0)
    ph = result.physical
    assert ph is not None
    assert ph.calibrated is True
    assert ph.relaxed_max_cm == 11.2
    assert ph.flexed_max_cm == 14.0
    assert abs(ph.change_max_cm - 2.8) < 1e-9
    assert abs(ph.change_mean_cm - 2.5) < 1e-9
    assert "not medical" in result.measurement_note


def test_physical_dimensions_not_calibrated_without_reference():
    relaxed = {"peak_bulge": 0.30, "definition": 0.05, "shape": 0.7, "curvature": 0.05,
               "max_width_cm": 11.2, "mean_width_cm": 10.0}
    flexed = {"peak_bulge": 0.50, "definition": 0.10, "shape": 0.8, "curvature": 0.10,
              "max_width_cm": 14.0, "mean_width_cm": 12.5}
    result = score_scan(relaxed, flexed)
    ph = result.physical
    assert ph is not None
    assert ph.calibrated is False
    assert ph.change_max_cm is None
    assert "No arm-length reference" in result.measurement_note


def test_physical_dimensions_none_values_never_invent_widths():
    relaxed = {"peak_bulge": 0.30, "definition": 0.05, "shape": 0.7, "curvature": 0.05}
    flexed = {"peak_bulge": 0.50, "definition": 0.10, "shape": 0.8, "curvature": 0.10}
    result = score_scan(relaxed, flexed, reference_cm=30.0)
    ph = result.physical
    assert ph is not None
    assert ph.calibrated is False
    assert ph.relaxed_max_cm is None
    assert ph.flexed_max_cm is None

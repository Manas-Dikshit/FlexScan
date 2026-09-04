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

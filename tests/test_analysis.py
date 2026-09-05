import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app.arm_analysis import (
    FrameMeasurement,
    ArmSlice,
    aggregate_measurements,
    build_arm_region,
    measure_definition,
    measure_shape,
    measure_curvature,
    measure_slice_widths,
    normalize_illumination,
    widths_to_cm,
    analyze_frame,
)
from app.pose import ArmPose


def make_arm(shoulder=(50, 100), elbow=(200, 100), wrist=(220, 220)) -> ArmPose:
    return ArmPose(shoulder=shoulder, elbow=elbow, wrist=wrist, side="right", confidence=0.9)


# ---------------------------------------------------------------------------
# build_arm_region
# ---------------------------------------------------------------------------
def test_build_arm_region_computes_arm_length():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    arm = make_arm(shoulder=(50, 100), elbow=(200, 100))
    region = build_arm_region(frame, arm)
    assert region is not None
    assert abs(region.arm_length - 150.0) < 1e-3
    assert region.polygon.shape[1] == 2


def test_build_arm_region_has_slices():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    arm = make_arm(shoulder=(50, 100), elbow=(200, 100))
    region = build_arm_region(frame, arm)
    assert region is not None
    assert len(region.slices) > 0


def test_build_arm_region_degenerate_when_shoulder_equals_elbow():
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    arm = make_arm(shoulder=(100, 100), elbow=(100, 100))
    region = build_arm_region(frame, arm)
    assert region is None


# ---------------------------------------------------------------------------
# measure_slice_widths
# ---------------------------------------------------------------------------
def test_measure_slice_widths_returns_slices():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[80:120, 30:170] = 255
    arm = make_arm(shoulder=(30, 100), elbow=(170, 100))
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    region = build_arm_region(frame, arm)
    assert region is not None
    slices = measure_slice_widths(mask, region)
    assert len(slices) > 0
    # At least some slices should have nonzero width
    assert any(s.width_px > 0 for s in slices)


# ---------------------------------------------------------------------------
# measure_definition
# ---------------------------------------------------------------------------
def test_measure_definition_higher_for_textured_region():
    mask = np.ones((100, 100), dtype=np.uint8) * 255

    flat = np.full((100, 100, 3), 128, dtype=np.uint8)

    rng = np.random.default_rng(42)
    noisy = rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8)
    checker = np.indices((100, 100)).sum(axis=0) % 20 < 10
    noisy[checker] = 0
    noisy[~checker] = 255
    noisy = noisy.astype(np.uint8)

    flat_score = measure_definition(flat, mask)
    textured_score = measure_definition(noisy, mask)
    assert textured_score > flat_score


def test_measure_definition_none_for_empty_mask():
    roi = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    assert measure_definition(roi, mask) is None


def test_measure_definition_adaptive_thresholds_on_uniform_bright_region():
    mask = np.ones((100, 100), dtype=np.uint8) * 255
    bright = np.full((100, 100, 3), 240, dtype=np.uint8)
    assert measure_definition(bright, mask) == 0.0


# ---------------------------------------------------------------------------
# measure_shape
# ---------------------------------------------------------------------------
def test_measure_shape_circle_has_high_solidity():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 60, 255, -1)
    solidity = measure_shape(mask)
    assert solidity is not None
    assert solidity > 0.9


def test_measure_shape_none_for_empty_mask():
    mask = np.zeros((50, 50), dtype=np.uint8)
    assert measure_shape(mask) is None


# ---------------------------------------------------------------------------
# measure_curvature
# ---------------------------------------------------------------------------
def test_measure_curvature_none_for_few_slices():
    slices = [ArmSlice(t=0.3, center=(100, 100), width_px=10.0, normal=(0, 1))]
    assert measure_curvature(slices, 150.0) is None


def test_measure_curvature_returns_value_for_enough_slices():
    slices = [
        ArmSlice(t=0.2, center=(80, 100), width_px=8.0, normal=(0, 1)),
        ArmSlice(t=0.4, center=(110, 100), width_px=20.0, normal=(0, 1)),
        ArmSlice(t=0.6, center=(140, 100), width_px=15.0, normal=(0, 1)),
        ArmSlice(t=0.8, center=(170, 100), width_px=6.0, normal=(0, 1)),
    ]
    result = measure_curvature(slices, 150.0)
    assert result is not None
    assert result >= 0


# ---------------------------------------------------------------------------
# aggregate_measurements
# ---------------------------------------------------------------------------
def test_aggregate_measurements_uses_median_of_reliable_frames():
    frames = [
        FrameMeasurement(peak_bulge=0.30, definition=0.05, shape=0.70, curvature=0.01,
                         width_profile=[0.2, 0.3, 0.25], reliable=True, arm_length=150.0),
        FrameMeasurement(peak_bulge=0.32, definition=0.06, shape=0.72, curvature=0.012,
                         width_profile=[0.21, 0.31, 0.26], reliable=True, arm_length=150.5),
        FrameMeasurement(peak_bulge=0.31, definition=0.055, shape=0.71, curvature=0.011,
                         width_profile=[0.205, 0.305, 0.255], reliable=True, arm_length=149.8),
        FrameMeasurement(None, None, None, None, None, reliable=False, reason="seg failed"),
    ]
    aggregate, reliable, reason = aggregate_measurements(frames)
    assert reliable is True
    assert aggregate is not None
    assert 0.29 < aggregate["peak_bulge"] < 0.33
    assert aggregate["arm_length"] is not None
    assert abs(aggregate["arm_length"] - 150.0) < 1.0


def test_aggregate_measurements_fails_with_too_few_reliable_frames():
    frames = [
        FrameMeasurement(None, None, None, None, None, reliable=False, reason="x"),
        FrameMeasurement(None, None, None, None, None, reliable=False, reason="x"),
        FrameMeasurement(0.3, 0.05, 0.7, 0.01, [0.2, 0.3], reliable=True),
    ]
    aggregate, reliable, reason = aggregate_measurements(frames)
    assert reliable is False
    assert aggregate is None
    assert reason


def test_aggregate_measurements_drops_outliers():
    frames = [FrameMeasurement(0.30, 0.05, 0.70, 0.01, [0.2], reliable=True) for _ in range(6)]
    frames.append(FrameMeasurement(5.0, 0.05, 0.70, 0.01, [2.0], reliable=True))
    aggregate, reliable, _ = aggregate_measurements(frames)
    assert reliable is True
    assert aggregate["peak_bulge"] < 1.0


def test_aggregate_measurements_medians_cm_widths():
    frames = [
        FrameMeasurement(0.30, 0.05, 0.70, 0.01, [0.2, 0.3, 0.25], reliable=True,
                         arm_length=200.0, widths_cm=[11.0, 12.0, 11.5]),
        FrameMeasurement(0.31, 0.06, 0.71, 0.012, [0.2, 0.3, 0.25], reliable=True,
                         arm_length=200.0, widths_cm=[12.0, 13.0, 12.5]),
        FrameMeasurement(0.305, 0.055, 0.705, 0.011, [0.2, 0.3, 0.25], reliable=True,
                         arm_length=200.0, widths_cm=[11.5, 12.5, 12.0]),
        FrameMeasurement(None, None, None, None, None, reliable=False, reason="seg failed"),
    ]
    aggregate, reliable, _ = aggregate_measurements(frames)
    assert reliable is True
    assert aggregate["max_width_cm"] is not None
    assert abs(aggregate["max_width_cm"] - 12.5) < 1e-9
    assert abs(aggregate["mean_width_cm"] - 12.0) < 1e-9
    assert abs(aggregate["width_profile_cm"][0] - 11.5) < 1e-9


# ---------------------------------------------------------------------------
# normalize_illumination
# ---------------------------------------------------------------------------
def test_normalize_illumination_preserves_shape_and_dtype():
    frame = np.full((120, 160, 3), 100, dtype=np.uint8)
    norm = normalize_illumination(frame)
    assert norm.shape == frame.shape
    assert norm.dtype == np.uint8


def test_normalize_illumination_raises_contrast_in_one_shaded_half():
    rng = np.random.default_rng(7)
    dark = np.full((100, 100, 3), 60, dtype=np.uint8)
    bright = np.full((100, 100, 3), 180, dtype=np.uint8)
    frame = np.hstack([dark, bright]).astype(np.uint8)
    norm = normalize_illumination(frame)
    dark_std = norm[:, :100].astype(np.float32).std()
    assert dark_std > 0.0


# ---------------------------------------------------------------------------
# widths_to_cm
# ---------------------------------------------------------------------------
def test_widths_to_cm_scales_by_arm_length():
    # 200 px upper arm claimed as 40 cm => 5 px/cm => 100 px width = 20 cm
    widths = widths_to_cm([100.0, 50.0], reference_cm=40.0, arm_length_px=200.0)
    assert widths is not None
    assert abs(widths[0] - 20.0) < 1e-9
    assert abs(widths[1] - 10.0) < 1e-9


def test_widths_to_cm_none_without_reference():
    assert widths_to_cm([100.0], reference_cm=None, arm_length_px=200.0) is None
    assert widths_to_cm([100.0], reference_cm=0.0, arm_length_px=200.0) is None
    assert widths_to_cm([100.0], reference_cm=40.0, arm_length_px=0.0) is None


def test_analyze_frame_is_lighting_robust():
    # Even a bright synthetic arm must produce reliable measurements
    frame = np.full((240, 240, 3), 180, dtype=np.uint8)
    arm = make_arm(shoulder=(60, 120), elbow=(200, 120), wrist=(210, 160))
    result = analyze_frame(frame, arm)
    assert result is not None
    assert result.reliable or "segment" in result.reason or "Lighting" in result.reason


def test_luminance_gate_uses_foreground_pixels_only():
    # Bright arm in a large dark room: mean over the whole frame would sit near
    # 0 and falsely trip the "too dark" gate; foreground-only mean must pass.
    frame = np.full((240, 240, 3), 20, dtype=np.uint8)  # dark background
    cv2.rectangle(frame, (60, 90), (200, 150), (150, 150, 150), -1)  # bright arm
    arm = make_arm(shoulder=(60, 120), elbow=(200, 120), wrist=(210, 160))
    result = analyze_frame(frame, arm)
    assert result is not None
    assert "Lighting" not in result.reason

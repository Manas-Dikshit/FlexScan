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

"""
Upper-arm region-of-interest extraction and visual feature measurement.

Uses shoulder-elbow geometry to define a polygon arm region, generates
dense perpendicular measurement slices along the arm axis, and computes
multiple features per slice for robust biceps analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app import config
from app.pose import ArmPose

Point = Tuple[float, float]


@dataclass
class ArmSlice:
    """One perpendicular cross-section measurement along the arm."""
    t: float                    # position along arm axis (0=shoulder, 1=elbow)
    center: Point               # midpoint on arm axis
    width_px: float             # foreground width at this slice
    normal: Point               # unit normal to the arm axis


@dataclass
class ArmRegion:
    """Polygon defining the analysis region around the upper arm."""
    polygon: np.ndarray         # (N, 2) float points defining the arm region
    shoulder: Point
    elbow: Point
    arm_length: float
    slices: List[ArmSlice] = field(default_factory=list)


@dataclass
class FrameMeasurement:
    """Single-frame measurements for one arm, based on dense slice analysis."""
    peak_bulge: Optional[float]
    definition: Optional[float]
    shape: Optional[float]
    curvature: Optional[float]
    width_profile: Optional[List[float]]  # normalized widths at each slice position
    reliable: bool
    reason: str = ""
    arm_length: Optional[float] = None   # shoulder-to-elbow pixels, for phase guards


# ---------------------------------------------------------------------------
# Arm region: polygon from shoulder-elbow geometry
# ---------------------------------------------------------------------------
def build_arm_region(frame: np.ndarray, arm: ArmPose) -> Optional[ArmRegion]:
    """
    Build a polygon arm region based on shoulder-elbow geometry.
    The polygon tapers from shoulder to elbow, providing a more anatomically
    accurate region than a simple rectangle.
    """
    h, w = frame.shape[:2]
    sx, sy = arm.shoulder
    ex, ey = arm.elbow
    arm_length = arm.upper_arm_length
    if arm_length < 1e-3:
        return None

    angle = arm.arm_angle_rad
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Arm axis direction and perpendicular normal
    axis_dx = cos_a
    axis_dy = sin_a
    norm_x = -sin_a
    norm_y = cos_a

    half_w = arm_length * config.ARM_POLYGON_WIDTH_FACTOR

    n_slices = config.NUM_MEASUREMENT_SLICES
    slices = []

    # Build polygon points: top edge shoulder->elbow, bottom edge elbow->shoulder
    top_pts = []
    bot_pts = []
    for i in range(n_slices + 1):
        t = i / n_slices
        # Position along axis
        px = sx + t * (ex - sx)
        py = sy + t * (ey - sy)

        # Taper width: wider at mid-bicep, narrower near shoulder/elbow
        # Parabolic taper peaking at t=0.45 (bicep belly location)
        taper = 1.0 - (2.0 * (t - 0.45)) ** 2
        taper = max(0.3, taper)
        w_at_t = half_w * taper

        top_pts.append((px + norm_x * w_at_t, py + norm_y * w_at_t))
        bot_pts.append((px - norm_x * w_at_t, py - norm_y * w_at_t))

        if i > 0 and i < n_slices:
            slices.append(ArmSlice(
                t=t,
                center=(px, py),
                width_px=0.0,
                normal=(norm_x, norm_y),
            ))

    polygon = np.array(top_pts + bot_pts[::-1], dtype=np.float32)

    # Clip polygon to frame bounds
    polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

    region = ArmRegion(
        polygon=polygon,
        shoulder=arm.shoulder,
        elbow=arm.elbow,
        arm_length=arm_length,
        slices=slices,
    )
    return region


# ---------------------------------------------------------------------------
# Segmentation: YOLOv8-seg model first, then GrabCut fallback
# ---------------------------------------------------------------------------
def segment_arm_region(frame: np.ndarray, region: ArmRegion) -> Optional[np.ndarray]:
    """
    Segment the arm using the YOLOv8-seg person model (if available),
    falling back to GrabCut when the model is not present.
    Returns a binary mask or None.
    """
    h, w = frame.shape[:2]
    if h < 8 or w < 8:
        return None

    # Try the segmentation model first
    try:
        from app.segmentation import segment_with_model
        model_mask = segment_with_model(frame, region.polygon)
        if model_mask is not None:
            return model_mask
    except ImportError:
        pass

    # Fallback: GrabCut within the arm polygon
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    pts = region.polygon.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(poly_mask, [pts], 255)

    if poly_mask.sum() == 0:
        return None

    gc_mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    gc_mask[poly_mask > 0] = cv2.GC_PR_FGD

    try:
        cv2.grabCut(frame, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None

    binary = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    binary = cv2.bitwise_and(binary, poly_mask)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if binary.sum() == 0:
        return None
    return binary


# ---------------------------------------------------------------------------
# Slice-based measurements
# ---------------------------------------------------------------------------
def measure_slice_widths(mask: np.ndarray, region: ArmRegion) -> List[ArmSlice]:
    """Measure foreground width at each slice position along the arm."""
    h, w = mask.shape[:2]
    sx, sy = region.shoulder
    ex, ey = region.elbow
    arm_length = region.arm_length
    if arm_length < 1e-3:
        return []

    angle = np.arctan2(ey - sy, ex - sx)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    norm_x = -sin_a
    norm_y = cos_a

    measured_slices = []
    for sl in region.slices:
        t = sl.t
        px = sx + t * (ex - sx)
        py = sy + t * (ey - sy)

        # Sample along the normal direction
        half_w = arm_length * config.ARM_POLYGON_WIDTH_FACTOR
        taper = 1.0 - (2.0 * (t - 0.45)) ** 2
        taper = max(0.3, taper)
        sample_range = half_w * taper * 1.2  # sample slightly beyond polygon

        # Count foreground pixels along the normal
        count = 0
        for dist in np.arange(-sample_range, sample_range, 1.0):
            sx_i = int(round(px + norm_x * dist))
            sy_i = int(round(py + norm_y * dist))
            if 0 <= sx_i < w and 0 <= sy_i < h and mask[sy_i, sx_i] > 0:
                count += 1

        measured_slice = ArmSlice(
            t=t,
            center=(px, py),
            width_px=float(count),
            normal=(norm_x, norm_y),
        )
        measured_slices.append(measured_slice)

    return measured_slices


def measure_definition(frame: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """Edge density inside the masked arm region, with lighting-adaptive thresholds."""
    if mask.sum() == 0:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    mean_lum = float(masked.mean())
    sigma = float(masked.std())
    lo = int(max(0.0, mean_lum - config.CANNY_EDGE_SIGMA * sigma))
    hi = int(min(255.0, mean_lum + config.CANNY_EDGE_SIGMA * sigma))
    if hi - lo < 1:
        hi = min(255, lo + 1)
    lo = max(1, lo)
    edges = cv2.Canny(gray, lo, hi)
    edges_in_mask = cv2.bitwise_and(edges, edges, mask=mask)
    mask_area = int((mask > 0).sum())
    if mask_area == 0:
        return None
    return float((edges_in_mask > 0).sum()) / mask_area


def measure_shape(mask: np.ndarray) -> Optional[float]:
    """Contour solidity from the segmented mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area <= 0:
        return None
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return None
    return float(area / hull_area)


def measure_curvature(slices: List[ArmSlice], arm_length: float) -> Optional[float]:
    """
    Measure how much the arm contour deviates from a straight line.
    Higher curvature = more bicep peak / definition.
    Computed from the width profile gradient changes.
    """
    if len(slices) < 3 or arm_length < 1e-3:
        return None
    widths = [s.width_px for s in slices]
    if all(w == 0 for w in widths):
        return None

    # Second derivative of width profile (curvature proxy)
    arr = np.array(widths, dtype=np.float64)
    # Normalize by arm length
    arr = arr / arm_length if arm_length > 0 else arr

    if arr.size < 3:
        return None
    second_deriv = np.diff(arr, n=2)
    # Curvature = mean absolute second derivative
    curvature = float(np.mean(np.abs(second_deriv)))
    return curvature


# ---------------------------------------------------------------------------
# Full per-frame pipeline
# ---------------------------------------------------------------------------
def analyze_frame(frame: np.ndarray, arm: ArmPose) -> FrameMeasurement:
    """Full per-frame pipeline: polygon region -> segmentation -> dense measurements."""
    region = build_arm_region(frame, arm)
    if region is None:
        return FrameMeasurement(
            peak_bulge=None, definition=None, shape=None, curvature=None,
            width_profile=None, reliable=False, reason="Arm region too small."
        )

    mask = segment_arm_region(frame, region)
    if mask is None:
        return FrameMeasurement(
            peak_bulge=None, definition=None, shape=None, curvature=None,
            width_profile=None, reliable=False, reason="Could not segment arm from background."
        )

    # Dense slice measurements
    slices = measure_slice_widths(mask, region)
    if not slices:
        return FrameMeasurement(
            peak_bulge=None, definition=None, shape=None, curvature=None,
            width_profile=None, reliable=False, reason="Could not compute slice measurements."
        )

    widths = [s.width_px for s in slices]
    if all(w == 0 for w in widths):
        return FrameMeasurement(
            peak_bulge=None, definition=None, shape=None, curvature=None,
            width_profile=None, reliable=False, reason="No foreground in measurement slices."
        )

    # Normalize widths by arm length
    arm_length = region.arm_length
    norm_widths = [w / arm_length for w in widths]
    peak_bulge = max(norm_widths)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_lum = float(cv2.bitwise_and(gray, gray, mask=mask).mean())
    if not (config.MIN_AVG_LUMINANCE <= roi_lum <= config.MAX_AVG_LUMINANCE):
        return FrameMeasurement(
            peak_bulge=None, definition=None, shape=None, curvature=None,
            width_profile=norm_widths, reliable=False,
            reason="Lighting is too dark or too bright -- reposition or adjust the light.",
            arm_length=arm_length,
        )

    definition = measure_definition(frame, mask)
    shape = measure_shape(mask)
    curvature = measure_curvature(slices, arm_length)

    if peak_bulge is None or definition is None or shape is None:
        return FrameMeasurement(
            peak_bulge=None, definition=None, shape=None, curvature=curvature,
            width_profile=norm_widths, reliable=False,
            reason="One or more features could not be computed."
        )

    return FrameMeasurement(
        peak_bulge=peak_bulge,
        definition=definition,
        shape=shape,
        curvature=curvature,
        width_profile=norm_widths,
        reliable=True,
    )


# ---------------------------------------------------------------------------
# Aggregation across frames
# ---------------------------------------------------------------------------
def aggregate_measurements(measurements: List[FrameMeasurement]) -> Tuple[Optional[dict], bool, str]:
    """Combine several FrameMeasurement objects using robust median."""
    reliable_frames = [m for m in measurements if m.reliable]
    min_needed = max(3, len(measurements) // 3)
    if len(reliable_frames) < min_needed:
        return None, False, "Too few stable, well-lit frames were captured."

    def robust_median(values):
        arr = np.array(values, dtype=np.float64)
        if arr.size >= 5:
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            filtered = arr[(arr >= lo) & (arr <= hi)]
            if filtered.size > 0:
                arr = filtered
        return float(np.median(arr))

    # Aggregate scalar features
    aggregated = {
        "peak_bulge": robust_median([m.peak_bulge for m in reliable_frames]),
        "definition": robust_median([m.definition for m in reliable_frames]),
        "shape": robust_median([m.shape for m in reliable_frames]),
        "curvature": robust_median([m.curvature for m in reliable_frames if m.curvature is not None]),
    }

    # Aggregate width profiles: median at each slice position
    profiles = [m.width_profile for m in reliable_frames if m.width_profile is not None]
    if profiles and all(len(p) == len(profiles[0]) for p in profiles):
        profile_arr = np.array(profiles, dtype=np.float64)
        aggregated["width_profile"] = np.median(profile_arr, axis=0).tolist()
    else:
        aggregated["width_profile"] = None

    return aggregated, True, ""

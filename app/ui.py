"""
Minimal OpenCV-window UI: draws the camera feed, full upper-body skeleton,
arm polygon overlay, dense measurement points, status, and clickable buttons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from app.arm_analysis import ArmRegion
from app.pose import PoseResult
from app.scoring import ScanResult
from app.state import ScanSession, ScanState

WHITE = (255, 255, 255)
GREEN = (80, 220, 120)
YELLOW = (60, 220, 240)
RED = (70, 70, 235)
GRAY = (120, 120, 120)
DARK_BG = (30, 30, 30)
CYAN = (220, 180, 60)
ORANGE = (50, 165, 255)
ACCENT = (110, 180, 90)
HEADER_BG = (24, 24, 38)


@dataclass
class Button:
    label: str
    x1: int
    y1: int
    x2: int
    y2: int
    callback: Callable[[], None]
    color: Tuple[int, int, int] = (90, 90, 90)

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def draw(self, frame: np.ndarray) -> None:
        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), self.color, -1)
        cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), WHITE, 1, cv2.LINE_AA)
        text_size = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        tx = self.x1 + (self.x2 - self.x1 - text_size[0]) // 2
        ty = self.y1 + (self.y2 - self.y1 + text_size[1]) // 2
        cv2.putText(frame, self.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)


class ButtonManager:
    def __init__(self):
        self.buttons: List[Button] = []

    def set_buttons(self, buttons: List[Button]) -> None:
        self.buttons = buttons

    def draw(self, frame: np.ndarray) -> None:
        for b in self.buttons:
            b.draw(frame)

    def handle_click(self, x: int, y: int) -> None:
        for b in self.buttons:
            if b.contains(x, y):
                b.callback()
                return


def _put_lines(frame, lines: List[Tuple[str, Tuple[int, int, int]]], x: int, y: int, line_h: int = 26) -> None:
    for text, color in lines:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        y += line_h


def draw_pose_overlay(frame: np.ndarray, pose: PoseResult, locked_side: Optional[str]) -> None:
    """Draw full upper-body skeleton and all detected landmarks."""
    if not pose.person_detected:
        return

    # Draw skeleton connections
    for p1, p2 in pose.get_skeleton_lines():
        pt1 = tuple(map(int, p1))
        pt2 = tuple(map(int, p2))
        cv2.line(frame, pt1, pt2, GRAY, 1, cv2.LINE_AA)

    # Draw locked arm prominently
    arms = pose.arms
    if locked_side and locked_side in arms:
        arms = {locked_side: arms[locked_side]}
    for arm in arms.values():
        s = tuple(map(int, arm.shoulder))
        e = tuple(map(int, arm.elbow))
        w = tuple(map(int, arm.wrist))
        cv2.line(frame, s, e, GREEN, 3, cv2.LINE_AA)
        cv2.line(frame, e, w, GRAY, 2, cv2.LINE_AA)
        for pt in (s, e, w):
            cv2.circle(frame, pt, 6, YELLOW, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 6, WHITE, 1, cv2.LINE_AA)

    # Draw all upper-body keypoints as small dots
    for name, pt in pose.upper_body.keypoints.items():
        if name in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                     "left_wrist", "right_wrist"):
            continue  # already drawn as arm joints
        ix, iy = int(pt[0]), int(pt[1])
        cv2.circle(frame, (ix, iy), 3, GRAY, -1, cv2.LINE_AA)


def draw_arm_region(frame: np.ndarray, region: ArmRegion) -> None:
    """Draw the arm polygon and dense measurement slice lines."""
    if region is None:
        return

    # Draw polygon outline
    pts = region.polygon.astype(np.int32)
    cv2.polylines(frame, [pts], True, CYAN, 1, cv2.LINE_AA)

    # Draw measurement slice lines (perpendicular to arm axis)
    for sl in region.slices:
        cx, cy = int(sl.center[0]), int(sl.center[1])
        nx, ny = sl.normal
        # Extend line on both sides
        ext = 25  # visual extension in pixels
        p1 = (int(cx + nx * ext), int(cy + ny * ext))
        p2 = (int(cx - nx * ext), int(cy - ny * ext))
        cv2.line(frame, p1, p2, ORANGE, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 2, ORANGE, -1, cv2.LINE_AA)


def draw_measurement_points(frame: np.ndarray, slices, arm_length: float) -> None:
    """Draw measurement points showing the width profile on the arm."""
    if not slices or arm_length < 1e-3:
        return
    for sl in slices:
        cx, cy = sl.center
        hw = sl.width_px / 2.0
        nx, ny = sl.normal
        # Draw width indicator dots at both edges
        p1 = (int(cx + nx * hw), int(cy + ny * hw))
        p2 = (int(cx - nx * hw), int(cy - ny * hw))
        cv2.circle(frame, p1, 3, GREEN, -1, cv2.LINE_AA)
        cv2.circle(frame, p2, 3, GREEN, -1, cv2.LINE_AA)
        cv2.circle(frame, p1, 3, WHITE, 1, cv2.LINE_AA)
        cv2.circle(frame, p2, 3, WHITE, 1, cv2.LINE_AA)


def draw_status(frame: np.ndarray, session: ScanSession, status_text: str) -> None:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 92), HEADER_BG, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    # Accent bar
    cv2.rectangle(frame, (0, 92), (w, 96), ACCENT, -1)

    cv2.putText(frame, "FlexScan", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, GREEN, 2, cv2.LINE_AA)
    cv2.putText(frame, "Visual Biceps Analysis", (130, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRAY, 1, cv2.LINE_AA)

    state_label = {
        ScanState.READY: "Position your arm inside the guide, then press [s]",
        ScanState.RELAXED_SCAN: "RELAXED SCAN - relax your arm and hold still",
        ScanState.RELAXED_CAPTURED: "Relaxed scan captured - press [f] to flex",
        ScanState.FLEX_SCAN: "NOW FLEX your bicep and hold still",
        ScanState.FLEX_CAPTURED: "Flexed scan captured - press [a] to analyze",
        ScanState.ANALYZE: "Analyzing...",
        ScanState.RESULT: "Scan complete",
    }.get(session.state, "")

    error_color = RED if session.last_error else WHITE
    error_text = session.last_error if session.last_error else status_text

    _put_lines(
        frame,
        [
            (f"{state_label}", ACCENT if session.last_error else WHITE),
            (error_text, error_color),
        ],
        x=16,
        y=74,
        line_h=24,
    )

    # Controls hint on the right of the header
    controls = "[s] Start   [f] Flex   [a] Analyze   [q] Quit"
    tsize = cv2.getTextSize(controls, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    tx = w - tsize[0] - 16
    cv2.putText(frame, controls, (tx, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1, cv2.LINE_AA)

    # Calibration status on the right, under the controls
    if session.reference_cm:
        cal = f"Cal: arm {session.reference_cm:.1f} cm - ESTIMATED"
    else:
        cal = "Cal: relative only (set arm length)"
    cv2.putText(frame, cal, (tx, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ACCENT if session.reference_cm else GRAY, 1, cv2.LINE_AA)


def draw_result(frame: np.ndarray, result: ScanResult) -> None:
    h, w = frame.shape[:2]
    panel_w = 360
    x0 = w - panel_w - 16
    y0 = 110
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (w - 16, h - 16), HEADER_BG, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (x0, y0), (w - 16, h - 16), ACCENT, 2, cv2.LINE_AA)

    # Panel header
    bar_h = 40
    cv2.rectangle(frame, (x0, y0), (w - 16, y0 + bar_h), ACCENT, -1)
    cv2.putText(frame, "SCAN RESULT", (x0 + 16, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)

    y = y0 + bar_h + 38
    if result.overall_score is not None:
        cv2.putText(frame, "VISUAL BICEPS SCORE", (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRAY, 1, cv2.LINE_AA)
        y += 44
        cv2.putText(frame, f"{result.overall_score:.1f}", (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, GREEN, 4, cv2.LINE_AA)
        cv2.putText(frame, "/ 10", (x0 + 130, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, GRAY, 2, cv2.LINE_AA)
        y += 52
    else:
        cv2.putText(frame, "Score unavailable", (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
        y += 44

    def fmt(v):
        return f"{v:.1f}" if v is not None else "n/a"

    # Estimated physical dimensions block
    if result.physical is not None:
        ph = result.physical
        y += 4
        if ph.calibrated:
            cv2.putText(frame, "ESTIMATED WIDTHS (cm)", (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT, 1, cv2.LINE_AA)
            y += 22
            phys_rows = [
                ("Max width   R/F", f"{fmt(ph.relaxed_max_cm)} / {fmt(ph.flexed_max_cm)}"),
                ("Avg width    R/F", f"{fmt(ph.relaxed_mean_cm)} / {fmt(ph.flexed_mean_cm)}"),
                ("Flex change (max)", f"{ph.change_max_cm:+.2f}" if ph.change_max_cm is not None else "n/a"),
                ("Flex change (avg)", f"{ph.change_mean_cm:+.2f}" if ph.change_mean_cm is not None else "n/a"),
            ]
            for lab, val in phys_rows:
                cv2.putText(frame, lab, (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
                tsize = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.putText(frame, val, (x0 + panel_w - 16 - tsize[0] - 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)
                y += 24
            method = f"Scale: your upper-arm length {ph.reference_cm:.1f} cm"
            cv2.putText(frame, method, (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1, cv2.LINE_AA)
            y += 24
        else:
            cv2.putText(frame, "No arm-length reference - relative only", (x0 + 16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1, cv2.LINE_AA)
            y += 24

    c = result.components
    rows = [
        ("Peak Bulge", fmt(c.peak_bulge)),
        ("Flex Response", fmt(c.flex_change)),
        ("Definition", fmt(c.definition)),
        ("Shape", fmt(c.shape)),
        ("Curvature", fmt(c.curvature)),
    ]
    for label, val in rows:
        cv2.putText(frame, label, (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        # value on the right, colored green if measurable
        vcolor = GREEN if val != "n/a" else GRAY
        tsize = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        cv2.putText(frame, val, (x0 + panel_w - 16 - tsize[0] - 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, vcolor, 1, cv2.LINE_AA)
        y += 26

    y += 10
    cv2.line(frame, (x0 + 16, y), (w - 32, y), (80, 80, 80), 1, cv2.LINE_AA)
    y += 18
    cv2.putText(frame, "Observations", (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GRAY, 1, cv2.LINE_AA)
    y += 26
    for line in result.advice:
        wrapped = _wrap_text(line, 36)
        for w_line in wrapped:
            cv2.putText(frame, f"- {w_line}", (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
            y += 22

    # Honesty note (estimated, not a medical measurement)
    if result.measurement_note:
        y += 6
        for w_line in _wrap_text(result.measurement_note, 40)[:4]:
            cv2.putText(frame, w_line, (x0 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1, cv2.LINE_AA)
            y += 16


def _wrap_text(text: str, max_chars: int) -> List[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines

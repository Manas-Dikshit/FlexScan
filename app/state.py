"""
Simple state machine driving the scan flow:

    READY -> RELAXED_SCAN -> RELAXED_CAPTURED -> FLEX_SCAN -> FLEX_CAPTURED
          -> ANALYZE -> RESULT

No extra states are introduced beyond what the flow actually needs.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional

import numpy as np

from app import config
from app.arm_analysis import FrameMeasurement, ArmRegion, analyze_frame, aggregate_measurements, build_arm_region
from app.pose import ArmPose, PoseResult


class ScanState(str, Enum):
    READY = "READY"
    RELAXED_SCAN = "RELAXED_SCAN"
    RELAXED_CAPTURED = "RELAXED_CAPTURED"
    FLEX_SCAN = "FLEX_SCAN"
    FLEX_CAPTURED = "FLEX_CAPTURED"
    ANALYZE = "ANALYZE"
    RESULT = "RESULT"


@dataclass
class ScanSession:
    state: ScanState = ScanState.READY
    arm_side: Optional[str] = None
    reference_cm: Optional[float] = (
        config.UPPER_ARM_LENGTH_CM if config.UPPER_ARM_LENGTH_CM > 0 else None
    )

    relaxed_measurements: List[FrameMeasurement] = field(default_factory=list)
    flexed_measurements: List[FrameMeasurement] = field(default_factory=list)

    relaxed_aggregate: Optional[dict] = None
    flexed_aggregate: Optional[dict] = None

    current_arm_region: Optional[ArmRegion] = None

    phase_started_at: float = field(default_factory=time.time)
    recent_arm_pose: Deque = field(default_factory=lambda: deque(maxlen=config.STABILITY_WINDOW))
    last_error: str = ""

    # -- lifecycle -----------------------------------------------------
    def start_scan(self) -> None:
        if self.state != ScanState.READY:
            return
        self._enter(ScanState.RELAXED_SCAN)

    def scan_again(self) -> None:
        self.__init__()

    # -- helpers ---------------------------------------------------------
    def _enter(self, new_state: ScanState) -> None:
        self.state = new_state
        self.phase_started_at = time.time()
        self.recent_arm_pose.clear()

    def _phase_timed_out(self) -> bool:
        return (time.time() - self.phase_started_at) > config.SCAN_TIMEOUT_SECONDS

    def is_arm_stable(self, arm: ArmPose) -> bool:
        self.recent_arm_pose.append((arm.elbow, arm.shoulder, arm.upper_arm_length))
        if len(self.recent_arm_pose) < config.STABILITY_WINDOW:
            return False
        elbows = np.array([e for e, _, _ in self.recent_arm_pose])
        shoulders = np.array([s for _, s, _ in self.recent_arm_pose])
        lengths = [l for _, _, l in self.recent_arm_pose]
        elbow_spread = np.linalg.norm(elbows.max(axis=0) - elbows.min(axis=0))
        shoulder_spread = np.linalg.norm(shoulders.max(axis=0) - shoulders.min(axis=0))
        length_jitter = (max(lengths) - min(lengths)) / max(1e-3, float(np.median(lengths)))
        return (
            elbow_spread <= config.STABILITY_TOLERANCE_PX
            and shoulder_spread <= config.STABILITY_TOLERANCE_PX * 1.5
            and length_jitter <= config.ARM_LENGTH_JITTER_FRACTION
        )

    def _select_arm(self, pose: PoseResult) -> Optional[ArmPose]:
        if self.arm_side and self.arm_side in pose.arms:
            return pose.arms[self.arm_side]
        if self.arm_side:
            return None
        return pose.best_arm

    # -- main per-frame update -------------------------------------------
    def update(self, frame, pose: PoseResult) -> str:
        if self.state in (ScanState.READY, ScanState.RESULT):
            return "Idle."

        if not pose.person_detected:
            self.last_error = "No person detected."
            return self.last_error

        arm = self._select_arm(pose)
        if arm is None:
            self.last_error = "Arm landmarks (shoulder/elbow/wrist) not clearly visible."
            return self.last_error
        self.last_error = ""

        if self.state == ScanState.RELAXED_SCAN:
            return self._run_capture_phase(frame, arm, self.relaxed_measurements, is_relaxed=True)

        if self.state == ScanState.FLEX_SCAN:
            return self._run_capture_phase(frame, arm, self.flexed_measurements, is_relaxed=False)

        return "Waiting..."

    def _run_capture_phase(self, frame, arm: ArmPose, buffer: List[FrameMeasurement], is_relaxed: bool) -> str:
        if self.arm_side is None:
            self.arm_side = arm.side

        stable = self.is_arm_stable(arm)
        if not stable:
            if self._phase_timed_out():
                self.last_error = "Timed out waiting for a stable arm position."
            return "Hold still..."

        if self._is_arm_length_outlier(buffer, arm):
            return f"Frames: {len(buffer)}/{config.REQUIRED_STABLE_FRAMES}"

        # Build arm region for visualization (update each frame for accuracy)
        self.current_arm_region = build_arm_region(frame, arm)

        if len(buffer) < config.REQUIRED_STABLE_FRAMES:
            measurement = analyze_frame(frame, arm)
            buffer.append(measurement)

        if len(buffer) >= config.REQUIRED_STABLE_FRAMES:
            if is_relaxed:
                self._finish_relaxed()
            else:
                self._finish_flexed()
            return "Captured."

        return f"Frames: {len(buffer)}/{config.REQUIRED_STABLE_FRAMES}"

    def _is_arm_length_outlier(self, buffer: List[FrameMeasurement], arm: ArmPose) -> bool:
        lengths = [m.arm_length for m in buffer if m.arm_length]
        if len(lengths) < 2:
            return False
        med = float(np.median(lengths))
        if med < 1e-3:
            return False
        return abs(arm.upper_arm_length - med) / med > config.ARM_LENGTH_JITTER_FRACTION

    def _finish_relaxed(self) -> None:
        aggregate, reliable, reason = aggregate_measurements(self.relaxed_measurements)
        self.relaxed_aggregate = aggregate
        if not reliable:
            self.last_error = reason
        self._enter(ScanState.RELAXED_CAPTURED)

    def _finish_flexed(self) -> None:
        aggregate, reliable, reason = aggregate_measurements(self.flexed_measurements)
        self.flexed_aggregate = aggregate
        if not reliable:
            self.last_error = reason
        self._enter(ScanState.FLEX_CAPTURED)

    # -- manual transitions driven by the UI loop -------------------------
    def proceed_to_flex_scan(self) -> None:
        if self.state == ScanState.RELAXED_CAPTURED:
            self._enter(ScanState.FLEX_SCAN)

    def proceed_to_analyze(self) -> None:
        if self.state == ScanState.FLEX_CAPTURED:
            self._enter(ScanState.ANALYZE)

    def finish_analysis(self) -> None:
        if self.state == ScanState.ANALYZE:
            self._enter(ScanState.RESULT)

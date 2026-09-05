"""
Pose / landmark detection.

Wraps a pretrained YOLOv8-pose model (downloaded by download_models.py)
via the `ultralytics` library. Extracts all upper-body COCO-17 keypoints
and builds a full upper-body skeleton representation alongside per-arm
joint data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app import config

Point = Tuple[float, float]


class ModelNotDownloadedError(RuntimeError):
    """Raised when the pose model file is missing from models/."""


@dataclass
class ArmPose:
    side: str
    shoulder: Point
    elbow: Point
    wrist: Point
    confidence: float

    @property
    def upper_arm_length(self) -> float:
        sx, sy = self.shoulder
        ex, ey = self.elbow
        return float(np.hypot(ex - sx, ey - sy))

    @property
    def arm_angle_rad(self) -> float:
        sx, sy = self.shoulder
        ex, ey = self.elbow
        return float(np.arctan2(ey - sy, ex - sx))


@dataclass
class UpperBodyLandmarks:
    """All detected upper-body keypoints, keyed by COCO name."""
    keypoints: Dict[str, Point] = field(default_factory=dict)
    confidences: Dict[str, float] = field(default_factory=dict)

    def get(self, name: str) -> Optional[Point]:
        return self.keypoints.get(name)

    def has(self, name: str) -> bool:
        return name in self.keypoints and self.confidences.get(name, 0) >= config.POSE_CONFIDENCE_THRESHOLD


@dataclass
class PoseResult:
    person_detected: bool
    arms: Dict[str, ArmPose]
    upper_body: UpperBodyLandmarks = field(default_factory=UpperBodyLandmarks)
    raw_keypoints: Optional[np.ndarray] = None

    @property
    def best_arm(self) -> Optional[ArmPose]:
        if not self.arms:
            return None
        return max(self.arms.values(), key=lambda a: a.confidence)

    def get_skeleton_lines(self) -> List[Tuple[Point, Point]]:
        """Return all drawable upper-body connections where both endpoints exist."""
        lines = []
        for a_name, b_name in config.UPPER_BODY_CONNECTIONS:
            pa = self.upper_body.get(a_name)
            pb = self.upper_body.get(b_name)
            if pa is not None and pb is not None:
                lines.append((pa, pb))
        return lines


class PoseDetector:
    """Loads the local YOLOv8-pose checkpoint and extracts arm joints."""

    def __init__(self, model_path=None):
        self.model_path = model_path or config.POSE_MODEL_PATH
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise ModelNotDownloadedError(
                f"Pose model not found at '{self.model_path}'. "
                "Run `python download_models.py` first (see README)."
            )
        from ultralytics import YOLO
        self._model = YOLO(str(self.model_path))

    def detect(self, frame: np.ndarray) -> PoseResult:
        """Run pose detection on a single BGR frame."""
        self._ensure_loaded()

        results = self._model.predict(
            frame, verbose=False, conf=config.PERSON_CONFIDENCE_THRESHOLD
        )
        if not results or results[0].keypoints is None or len(results[0].boxes) == 0:
            return PoseResult(person_detected=False, arms={})

        result = results[0]
        box_confs = result.boxes.conf.cpu().numpy()
        kpts_all = result.keypoints.xy.cpu().numpy()
        kpts_conf_all = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None \
            else None

        arm_joint_idx = sorted({
            config.KEYPOINT_INDEX[name]
            for joints in config.REQUIRED_JOINTS_PER_ARM.values()
            for name in joints
        })

        best_idx = 0
        best_score = -1.0
        for i in range(kpts_all.shape[0]):
            confs = kpts_conf_all[i] if kpts_conf_all is not None else np.ones(kpts_all.shape[1])
            arm_conf = float(np.mean(confs[arm_joint_idx]))
            score = float(box_confs[i]) * max(arm_conf, 1e-4)
            if score > best_score:
                best_score = score
                best_idx = i

        kpts_xy = kpts_all[best_idx]
        kpts_conf = kpts_conf_all[best_idx] if kpts_conf_all is not None \
            else np.ones(kpts_xy.shape[0])
        raw = np.concatenate([kpts_xy, kpts_conf[:, None]], axis=1)

        upper_body = UpperBodyLandmarks()
        for name, idx in config.KEYPOINT_INDEX.items():
            x, y, c = raw[idx]
            upper_body.keypoints[name] = (float(x), float(y))
            upper_body.confidences[name] = float(c)

        arms: Dict[str, ArmPose] = {}
        for side, joint_names in config.REQUIRED_JOINTS_PER_ARM.items():
            joints = []
            confs = []
            usable = True
            for name in joint_names:
                idx = config.KEYPOINT_INDEX[name]
                x, y, c = raw[idx]
                if c < config.POSE_CONFIDENCE_THRESHOLD:
                    usable = False
                joints.append((float(x), float(y)))
                confs.append(float(c))
            if not usable:
                continue
            shoulder, elbow, wrist = joints
            arm = ArmPose(
                side=side,
                shoulder=shoulder,
                elbow=elbow,
                wrist=wrist,
                confidence=float(np.mean(confs)),
            )
            if arm.upper_arm_length < config.MIN_ARM_LENGTH_PX:
                continue
            arms[side] = arm

        return PoseResult(person_detected=True, arms=arms, upper_body=upper_body, raw_keypoints=raw)

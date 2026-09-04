"""
Lightweight person segmentation using YOLOv8-seg.

When the segmentation model is available, it provides a cleaner person mask
that is intersected with the arm polygon for more reliable arm isolation.
Falls back to GrabCut-based segmentation when the model is not present.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app import config


class SegmentationModel:
    """Lazy-loaded YOLOv8-seg for person segmentation."""

    def __init__(self):
        self._model = None
        self._available = None  # None = not checked yet

    def _ensure_loaded(self) -> bool:
        if self._available is not None:
            return self._available
        if not config.SEG_MODEL_PATH.exists():
            self._available = False
            return False
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(config.SEG_MODEL_PATH))
            self._available = True
            return True
        except Exception:
            self._available = False
            return False

    def segment_person(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run person segmentation on the frame.
        Returns a binary mask (uint8, 0/255) of the largest person, or None.
        """
        if not self._ensure_loaded():
            return None

        try:
            results = self._model.predict(frame, verbose=False, classes=[0])  # class 0 = person
        except Exception:
            return None

        if not results or results[0].masks is None:
            return None

        result = results[0]
        if len(result.boxes) == 0:
            return None

        # Pick the largest person mask
        h, w = frame.shape[:2]
        best_mask = None
        best_area = 0

        for i in range(len(result.boxes)):
            mask_data = result.masks.data[i].cpu().numpy()
            # Resize mask to frame size
            mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_LINEAR)
            binary = (mask_resized > 0.5).astype(np.uint8) * 255
            area = binary.sum() // 255
            if area > best_area:
                best_area = area
                best_mask = binary

        return best_mask


# Module-level singleton
_seg_model: Optional[SegmentationModel] = None


def get_seg_model() -> SegmentationModel:
    global _seg_model
    if _seg_model is None:
        _seg_model = SegmentationModel()
    return _seg_model


def segment_with_model(frame: np.ndarray, arm_polygon: np.ndarray) -> Optional[np.ndarray]:
    """
    Use YOLOv8-seg to get a person mask, then intersect with the arm polygon.
    Returns a binary mask or None if the model is unavailable/failed.
    """
    seg = get_seg_model()
    person_mask = seg.segment_person(frame)
    if person_mask is None:
        return None

    h, w = frame.shape[:2]
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    pts = arm_polygon.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(poly_mask, [pts], 255)

    # Intersect person mask with arm polygon
    arm_mask = cv2.bitwise_and(person_mask, poly_mask)

    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    arm_mask = cv2.morphologyEx(arm_mask, cv2.MORPH_OPEN, kernel)
    arm_mask = cv2.morphologyEx(arm_mask, cv2.MORPH_CLOSE, kernel)

    if arm_mask.sum() == 0:
        return None
    return arm_mask

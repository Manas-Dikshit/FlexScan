"""
Lightweight, lighting-robust person/arm segmentation.

Pipeline preference (best quality first):
  1. MODNet portrait matting (ONNX, Apache-2.0, Hugging Face) - soft arm
     boundary that is far less dependent on raw lighting than colour
     thresholds, and gives a cleaner edge than GrabCut.
  2. YOLOv8-seg person mask intersected with the arm polygon.
  3. OpenCV GrabCut seeded with the arm polygon (always available).

The matting model runs on an illumination-normalized frame, so shadows or
backlight do not wreck the boundary estimate.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from app import config


class MattingModel:
    """Lazy-loaded MODNet portrait-matting model via onnxruntime."""

    def __init__(self):
        self._session = None
        self._available = None  # None = not checked yet

    def _ensure_loaded(self) -> bool:
        if self._available is not None:
            return self._available
        if not config.MATTING_MODEL_PATH.exists():
            self._available = False
            return False
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                str(config.MATTING_MODEL_PATH),
                providers=["CPUExecutionProvider"],
            )
            self._available = True
        except Exception:
            self._available = False
        return self._available

    def matte_crop(self, frame: np.ndarray, box: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Return a full-frame soft alpha (float32, 0..1) of the foreground.
        If `box` (x, y, w, h) is given, only that region is matted so the
        subject fills the model input. None on failure.
        """
        if not self._ensure_loaded():
            return None
        h, w = frame.shape[:2]
        size = config.MATTING_INPUT_SIZE

        crop = frame
        ox = oy = 0
        cw, ch = w, h
        if box is not None:
            bx, by, bw0, bh0 = box
            pad_x = int(bw0 * 0.2)
            pad_y = int(bh0 * 0.2)
            ox = max(0, bx - pad_x)
            oy = max(0, by - pad_y)
            x2 = min(w, bx + bw0 + pad_x)
            y2 = min(h, by + bh0 + pad_y)
            if x2 - ox < 16 or y2 - oy < 16:
                return None
            crop = frame[oy:y2, ox:x2]
            cw, ch = crop.shape[1], crop.shape[0]

        try:
            resized = cv2.resize(crop, (size, size))
            blob = resized.astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)[None]
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
            blob = (blob - mean) / std
            out = self._session.run(None, {self._input_name(): blob})[0][0, 0]
        except Exception:
            return None

        if out.min() < 0.0 or out.max() > 1.0:
            out = 1.0 / (1.0 + np.exp(-out))
        alpha = np.clip(out, 0.0, 1.0)
        alpha = cv2.resize(alpha, (cw, ch), interpolation=cv2.INTER_LINEAR)

        full = np.zeros((h, w), dtype=np.float32)
        full[oy:oy + ch, ox:ox + cw] = alpha
        return full

    def _input_name(self) -> str:
        inp = self._session.get_inputs()[0]
        return inp.name


# Module-level singleton
_matting_model: Optional[MattingModel] = None


def get_matting_model() -> MattingModel:
    global _matting_model
    if _matting_model is None:
        _matting_model = MattingModel()
    return _matting_model


def _clean_arm_mask(person_mask: np.ndarray, arm_polygon: np.ndarray) -> Optional[np.ndarray]:
    """Intersect a person mask (uint8, 0/255) with the arm polygon and clean it up."""
    h, w = person_mask.shape[:2]
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    pts = arm_polygon.astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(poly_mask, [pts], 255)

    arm_mask = cv2.bitwise_and(person_mask, poly_mask)
    kernel = np.ones((3, 3), np.uint8)
    arm_mask = cv2.morphologyEx(arm_mask, cv2.MORPH_OPEN, kernel)
    arm_mask = cv2.morphologyEx(arm_mask, cv2.MORPH_CLOSE, kernel)

    if arm_mask.sum() == 0:
        return None
    return arm_mask


def segment_with_matting(
    frame: np.ndarray,
    arm_polygon: np.ndarray,
    person_box: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[np.ndarray]:
    """
    Use MODNet to estimate the foreground, then intersect with the arm polygon.
    Returns a binary arm mask or None if the model is unavailable/failed.
    """
    alpha = get_matting_model().matte_crop(frame, person_box)
    if alpha is None:
        return None
    binary = (alpha > config.MATTING_THRESHOLD).astype(np.uint8) * 255
    return _clean_arm_mask(binary, arm_polygon)


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
    person_mask = get_seg_model().segment_person(frame)
    if person_mask is None:
        return None
    return _clean_arm_mask(person_mask, arm_polygon)
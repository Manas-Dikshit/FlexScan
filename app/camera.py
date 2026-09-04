"""
Thin, error-handling wrapper around cv2.VideoCapture.

Keeps all webcam-specific error handling in one place so the rest of the
app never has to think about OpenCV capture quirks.
"""

from __future__ import annotations

import cv2

from app import config


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or read from."""


class Camera:
    def __init__(self, index: int = config.CAMERA_INDEX):
        self.index = index
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            self._cap.release()
            self._cap = None
            raise CameraError(
                f"Could not open webcam at index {self.index}. "
                "It may be unavailable or already in use by another application."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    def read(self):
        if self._cap is None:
            raise CameraError("Camera.read() called before open().")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise CameraError(
                "Failed to read a frame from the webcam. It may have been "
                "disconnected or is being used by another application."
            )
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

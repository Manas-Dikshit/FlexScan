"""
Downloads the FlexScan models (pose + optional segmentation) via Ultralytics.

Usage:
    python download_models.py
    python main.py

Models are downloaded from Ultralytics' servers (no token needed) and
saved to the models/ directory. Model files are git-ignored.
"""

from __future__ import annotations

import sys

from app import config


def main() -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[FlexScan] ultralytics is not installed. Run: pip install -r requirements.txt")
        return 1

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Pose model (required) ---
    if not config.POSE_MODEL_PATH.exists():
        print(f"[FlexScan] Downloading '{config.MODEL_FILENAME}'...")
        try:
            YOLO(str(config.POSE_MODEL_PATH))
        except Exception as e:
            print(f"[FlexScan] Pose model download failed: {e}")
            return 1
        if not config.POSE_MODEL_PATH.exists():
            print(f"[FlexScan] Pose model not found after download. Check config.")
            return 1
    print(f"[FlexScan] Pose model ready: {config.POSE_MODEL_PATH}")

    # --- Segmentation model (optional, improves accuracy) ---
    if not config.SEG_MODEL_PATH.exists():
        print(f"[FlexScan] Downloading '{config.SEG_MODEL_FILENAME}' (optional, improves arm isolation)...")
        try:
            YOLO(str(config.SEG_MODEL_PATH))
        except Exception as e:
            print(f"[FlexScan] Segmentation model download failed (optional): {e}")
            print("[FlexScan] Continuing without segmentation model -- GrabCut will be used instead.")
    else:
        print(f"[FlexScan] Segmentation model ready: {config.SEG_MODEL_PATH}")

    print("[FlexScan] You can now run: python main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

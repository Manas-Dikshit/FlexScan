"""
Downloads the FlexScan models (pose + optional segmentation + matting).

Usage:
    python download_models.py
    python main.py

Model sources and licenses:
    - yolov8n-pose.pt    -- Ultralytics (AGPL-3.0), required
    - yolov8n-seg.pt     -- Ultralytics (AGPL-3.0), optional arm isolation
    - modnet_photographic.onnx -- MOdesNet portrait matting, Apache-2.0,
      hosted on Hugging Face (Xenova/modnet). Improves the arm boundary
      under varied lighting; FlexScan falls back to the seg model, then
      GrabCut, if it is missing.

Model files are saved to the models/ directory and are git-ignored.
"""

from __future__ import annotations

import sys

from app import config


def _download(name: str, url: str, dest) -> bool:
    """Download a model file with a plain HTTP GET (no token needed)."""
    print(f"[FlexScan] Downloading '{name}' from {url} ...")
    try:
        import requests
    except ImportError:
        print("[FlexScan] requests is not installed. Run: pip install -r requirements.txt")
        return False
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
    except Exception as e:
        print(f"[FlexScan] Download failed: {e}")
        return False
    if not dest.exists() or dest.stat().st_size < 1024 * 1024:
        print("[FlexScan] Downloaded file is missing or too small. Check config.")
        return False
    return True


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

    # --- Segmentation model (optional, improves arm isolation) ---
    if not config.SEG_MODEL_PATH.exists():
        print(f"[FlexScan] Downloading '{config.SEG_MODEL_FILENAME}' (optional, improves arm isolation)...")
        try:
            YOLO(str(config.SEG_MODEL_PATH))
        except Exception as e:
            print(f"[FlexScan] Segmentation model download failed (optional): {e}")
            print("[FlexScan] Continuing without segmentation model -- GrabCut will be used instead.")
    else:
        print(f"[FlexScan] Segmentation model ready: {config.SEG_MODEL_PATH}")

    # --- Matting model (optional, lighting-robust person/arm separation) ---
    if not config.MATTING_MODEL_PATH.exists():
        ok = _download(config.MATTING_MODEL_FILENAME, config.MATTING_MODEL_URL, config.MATTING_MODEL_PATH)
        if ok:
            print(f"[FlexScan] Matting model ready: {config.MATTING_MODEL_PATH}")
            print(f"[FlexScan]   License: {config.MATTING_MODEL_LICENSE} (see Xenova/modnet on Hugging Face)")
        else:
            print("[FlexScan] Matting model unavailable -- seg model or GrabCut will be used instead.")
    else:
        print(f"[FlexScan] Matting model ready: {config.MATTING_MODEL_PATH}")

    print("[FlexScan] You can now run: python main.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

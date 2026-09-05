"""
Central configuration for FlexScan.

Keep every tunable value here so nothing is scattered through the codebase
as a magic number.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Model sources
# ---------------------------------------------------------------------------
MODEL_FILENAME = "yolov8n-pose.pt"
SEG_MODEL_FILENAME = "yolov8n-seg.pt"
POSE_MODEL_PATH = MODELS_DIR / MODEL_FILENAME
SEG_MODEL_PATH = MODELS_DIR / SEG_MODEL_FILENAME

# ---------------------------------------------------------------------------
# Pose detection
# ---------------------------------------------------------------------------
POSE_CONFIDENCE_THRESHOLD = 0.5
PERSON_CONFIDENCE_THRESHOLD = 0.4

KEYPOINT_INDEX = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

REQUIRED_JOINTS_PER_ARM = {
    "left": ("left_shoulder", "left_elbow", "left_wrist"),
    "right": ("right_shoulder", "right_elbow", "right_wrist"),
}

UPPER_BODY_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
]

# ---------------------------------------------------------------------------
# Scan / capture behaviour
# ---------------------------------------------------------------------------
REQUIRED_STABLE_FRAMES = 10
STABILITY_WINDOW = 5
STABILITY_TOLERANCE_PX = 12.0
ARM_LENGTH_JITTER_FRACTION = 0.15  # max relative arm-length swing considered stable
MAX_PHASE_GEOMETRY_CHANGE = 0.35   # max relative arm-length change between relaxed/flexed
SCAN_TIMEOUT_SECONDS = 25
MIN_ARM_LENGTH_PX = 60

# ---------------------------------------------------------------------------
# Measurement robustness
# ---------------------------------------------------------------------------
MIN_AVG_LUMINANCE = 55      # arm ROI mean gray level; below = too dark
MAX_AVG_LUMINANCE = 235     # arm ROI mean gray level; above = overexposed
CANNY_EDGE_SIGMA = 2.5      # auto edge thresholds: mean +/- sigma * std of arm ROI

# ---------------------------------------------------------------------------
# Arm region of interest
# ---------------------------------------------------------------------------
ROI_WIDTH_FACTOR = 0.9
ROI_MARGIN_FACTOR = 0.12

# ---------------------------------------------------------------------------
# Dense arm measurements
# ---------------------------------------------------------------------------
NUM_MEASUREMENT_SLICES = 7       # perpendicular cross-sections along arm
ARM_POLYGON_WIDTH_FACTOR = 0.45  # half-width of arm polygon as fraction of arm length
ARM_POLYGON_SHOULDER_OFFSET = 0.05
ARM_POLYGON_ELBOW_OFFSET = 0.05

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
SCORING_WEIGHTS = {
    "peak_bulge": 0.25,
    "flex_change": 0.30,
    "definition": 0.20,
    "shape": 0.15,
    "curvature": 0.10,
}

NORMALIZATION_RANGES = {
    "peak_bulge": (0.28, 0.62),
    "flex_change": (0.0, 0.35),
    "definition": (0.02, 0.18),
    "shape": (0.55, 0.95),
    "curvature": (0.0, 0.40),
}

SCORE_MIN = 1.0
SCORE_MAX = 10.0

# ---------------------------------------------------------------------------
# Camera / UI
# ---------------------------------------------------------------------------
CAMERA_INDEX = int(os.environ.get("FLEXSCAN_CAMERA_INDEX", "0"))
FRAME_WIDTH = 1280
FRAME_HEIGHT = 800
WINDOW_NAME = "FlexScan — Visual Biceps Analysis"

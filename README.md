# FlexScan — AI Biceps Visual Analysis

## Project overview

FlexScan is a local computer-vision application that watches your upper arm
through a webcam, first **relaxed** and then **flexed**, and produces a
**Visual Biceps Score (1–10)** with a few plain-language observations.

**Important:** FlexScan is a *visual, computer-vision estimate* built from
things a webcam can actually see — contour width, edge contrast, shape
compactness, contour curvature, and the change between relaxed and flexed
frames. It does **not** measure muscle mass, muscle fiber composition, muscle
architecture, body-fat percentage, exact arm circumference, or anything
medical/genetic, and it is not a fitness or medical diagnostic tool.

## Features

- Live webcam feed with full upper-body skeleton landmarks and analysis-region
  polygon overlay
- Dense per-slice arm measurements along the shoulder–elbow axis
- Guided two-phase scan: relaxed capture, then flexed capture
- Automatic checks for person/arm detection, landmark confidence, and frame
  stability before any frame is used
- Multi-frame capture (10 frames per phase) aggregated with robust median /
  outlier removal
- Five measured components: **Peak Bulge**, **Flex Response**, **Definition**,
  **Shape**, **Curvature**, combined into one weighted 1–10 score
- 1–3 deterministic, rule-based observations (no LLM)
- Runs fully locally after the one-time model download; nothing is uploaded

## Architecture

```text
Webcam
  → Pose Detection        (YOLOv8-pose, COCO-17 keypoints)
  → Arm Region             (polygon from shoulder→elbow geometry)
  → Segmentation           (YOLOv8-seg when available, else GrabCut)
  → Dense Slice Analysis    (7 perpendicular measurements along the arm)
  → Relaxed Scan            (10 stable frames captured)
  → Flexed Scan              (10 stable frames captured)
  → Comparison                 (relaxed vs. flexed across slice profile)
  → Score                       (weighted 1–10 Visual Biceps Score)
```

Module layout:

```text
FlexScan/
├── app/
│   ├── camera.py            webcam open/read/release, error handling
│   ├── pose.py               YOLOv8-pose wrapper -> full upper-body skeleton
│   │                          + shoulder/elbow/wrist per arm
│   ├── arm_analysis.py        arm polygon, dense slice measurement, feature math
│   ├── segmentation.py        YOLOv8-seg person mask (falls back to GrabCut)
│   ├── scoring.py              normalization, weighting, rule-based advice
│   ├── state.py                 scan state machine + stability/timeout logic
│   ├── ui.py                     OpenCV overlay drawing + clickable buttons
│   └── config.py                  every tunable value lives here
├── models/                 downloaded models (git-ignored)
├── download_models.py    model download script (via Ultralytics)
├── main.py               application entry point
└── tests/                 unit tests (no webcam required)
```

## Requirements

- Python 3.11+
- A webcam
- Optional GPU (used automatically if available; CPU works fine otherwise)
- Internet connection is only required once, to download the models

## Installation

```bash
git clone <repo>
cd FlexScan

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

(On macOS/Linux, activate with `source .venv/bin/activate` instead.)

## Model download

FlexScan downloads two models on first run via the Ultralytics package (no
token or account needed):

- **yolov8n-pose.pt** — required, detects the upper-body keypoints
- **yolov8n-seg.pt** — optional, provides a cleaner person mask for more
  reliable arm isolation. When absent, FlexScan falls back to OpenCV GrabCut.

```bash
python download_models.py
```

Models are saved to `models/` and git-ignored. To change the filenames,
update `MODEL_FILENAME` / `SEG_MODEL_FILENAME` in `app/config.py`.

## Running

```bash
python main.py
```

## How to use

```text
Press 's' (or click "Start Scan")
  → Relax your arm, hold still until the relaxed frames are captured
  → Press 'f' (or click "Continue")
  → Flex your bicep, hold still until the flexed frames are captured
  → Press 'a' (or click "Analyze")
  → View your Visual Biceps Score and observations
  → Press 's' again (or "Scan Again") to repeat, or 'q' to quit
```

Keyboard shortcuts: `s` start / scan again, `f` continue to flex, `a` analyze,
`q` quit.

## Landmark and measurement approach

### Pose landmarks

YOLOv8-pose returns the standard COCO-17 keypoints. FlexScan uses all
upper-body landmarks: nose, eyes, ears, shoulders, elbows, wrists, and hips.
All detected landmarks are drawn on the webcam feed as small dots, and the
upper-body skeleton connections (shoulder–shoulder, shoulder–elbow,
elbow–wrist, shoulder–hip, hip–hip) are drawn as connecting lines. The
actively analyzed arm is highlighted with a thicker, colored skeleton.

### Arm region polygon

Instead of a fixed rectangular crop, the upper-arm analysis region is a
**tapered polygon** derived from the shoulder–elbow geometry. The polygon is
built around the arm axis (shoulder → elbow) and tapers from a wider
mid-bicep belly toward the narrower shoulder and elbow ends, giving a more
anatomically accurate region. The polygon is shown in real time on the feed.

### Dense measurement slices

Inside the arm region, multiple **perpendicular cross-section lines** are
generated along the shoulder–elbow axis. Each slice measures the foreground
width of the arm at that position, and these are shown as orange measurement
lines with green edge dots during scanning. These slice widths are normalized
by arm length (so camera distance doesn't skew the result) and used to derive:

- **Peak Bulge** — the maximum normalized arm width (the bicep belly)
- **Curvature** — how sharply the width profile changes along the arm
- **Width profile** — the full set of widths used for comparison

### Segmentation

The arm is segmented from the background using a YOLOv8-seg person mask
intersected with the arm polygon. When the segmentation model is not
available, OpenCV GrabCut is used as a fallback, seeded with the polygon
region.

## Scoring methodology

Each captured frame is analyzed across the dense slice measurements:

- **Peak Bulge** — the maximum cross-section width of the segmented arm,
  normalized by shoulder-to-elbow length
- **Flex Response** — the change in peak bulge (and the full width profile)
  between flexed and relaxed captures; profile-based change boosts robustness
- **Definition** — edge/contrast density inside the segmented arm region
- **Shape** — contour solidity (area vs. convex-hull area) of the flexed arm
- **Curvature** — mean absolute second derivative of the normalized width
  profile, capturing the bicep peak's sharpness

Ten frames are captured per phase; each feature is aggregated across frames
using the median after discarding statistical outliers, so a single bad frame
can't swing the result. Width profiles are also medians across frames at each
slice position.

Each raw measurement is mapped onto a 1–10 scale using fixed, documented
ranges (`app/config.py: NORMALIZATION_RANGES`), then combined with these
weights into the overall score:

| Component       | Weight |
|------------------|-------:|
| Peak Bulge        |    25% |
| Flex Response      |   30% |
| Definition          |  20% |
| Shape/Contour        | 15% |
| Curvature           |  10% |

If a component cannot be measured reliably (e.g. bad segmentation), it is
excluded and the remaining weights are renormalized. If too much is
unreliable, FlexScan shows **"Unable to obtain a reliable scan — adjust
position/lighting and try again."** instead of producing a misleading score.

**These weights and ranges are heuristic engineering choices for a visual
computer-vision demo — they are not medical or scientific standards.**

## Troubleshooting

| Problem | What to check |
|---|---|
| Camera not opening | Make sure no other app is using the webcam, and that the OS has granted camera permission. Try a different `FLEXSCAN_CAMERA_INDEX` if you have multiple cameras. |
| "Pose model not found" | Run `python download_models.py` before `python main.py`. |
| Download failure | Check your internet connection. If the model name has changed, update the filenames in `app/config.py`. |
| Poor lighting / low definition score | Use even, diffuse lighting facing your arm; avoid strong backlight or deep shadows. |
| Arm not detected | Make sure your shoulder, elbow, and wrist are all visible in frame, and step back so the whole upper arm is in view. |
| Low FPS | Close other apps using the camera/GPU; a smaller `yolov8n` model already runs on CPU, but a GPU will speed it up further. |

## Privacy

All webcam processing happens **locally** on your machine. FlexScan does
not upload video, images, or measurements anywhere — the only network
access it uses is the one-time model download in `download_models.py`.

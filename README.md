# FlexScan - AI Biceps Visual Analysis

## Project overview

FlexScan is a local computer-vision application that watches your upper arm
through a webcam, first **relaxed** and then **flexed**, and produces a
**Visual Biceps Score (1-10)** with a few plain-language observations.

**Important:** FlexScan is a *visual, computer-vision estimate* built from
things a webcam can actually see - contour width, edge contrast, shape
compactness, contour curvature, and the change between relaxed and flexed
frames. It does **not** measure muscle mass, muscle fiber composition, muscle
architecture, body-fat percentage, exact arm circumference, or anything
medical or genetic, and it is not a fitness or medical diagnostic tool. Any
width shown in centimetres is an **estimated visual measurement** scaled from
pixels and your reference arm length - never an exact value. Scans that cannot
be estimated honestly are rejected rather than faked.

## Features

- Live webcam feed with full upper-body skeleton landmarks and analysis-region
  polygon overlay
- Dense per-slice arm measurements along the shoulder-elbow axis
- Guided two-phase scan: relaxed capture, then flexed capture
- Robust person selection: the tracked person is chosen by arm-landmark
  confidence, not just the raw detection-box score
- Stability gate that watches the shoulder, elbow, and arm length together, so
  small movements and keypoint jitter cannot corrupt a scan
- Multi-frame capture (10 frames per phase) aggregated with robust median and
  statistical outlier removal; frames with keypoint jumps are discarded
- Lighting tolerance: the frame is first run through illumination
  normalization (CLAHE), edge thresholds adapt to the arm region's brightness,
  and frames that remain too dark or overexposed are rejected with a clear
  reposition message
- Segmentation robustness: the arm is isolated by **MODNet portrait matting**
  (Apache-2.0, Hugging Face) for a soft, lighting-stable boundary, with
  YOLOv8-seg and then OpenCV GrabCut as automatic fallbacks
- **Estimated physical widths**: when you provide your upper-arm length in cm,
  per-slice pixel widths are converted to centimetres and shown honestly as
  estimates (max / average, relaxed vs. flexed, flex change); without a
  reference length only relative scores are shown - nothing is fabricated
- Distance-normalized measurements: every width is divided by the
  shoulder-to-elbow arm length, so camera distance and arm size do not skew
  the result
- Five measured components: **Peak Bulge**, **Flex Response**, **Definition**,
  **Shape**, **Curvature**, combined into one weighted 1-10 score
- Flex Response is the median of two real measurements (peak-slice change and
  full width-profile change); no invented or boosted values are ever used
- 1-3 deterministic, rule-based observations (no LLM)
- Unreliable scans are refused with guidance instead of producing a misleading
  score
- Runs fully locally after the one-time model download; nothing is uploaded
- No LLM, no cloud, no custom training - all models are permissive-licensed
  pretrained weights run on your machine

## Architecture

```text
Webcam
  -> Pose Detection          (YOLOv8-pose, COCO-17 keypoints)
  -> Person Selection        (highest combined box + arm-joint confidence)
  -> Arm Region              (tapered polygon from shoulder-elbow geometry)
  -> Illumination Normalize  (CLAHE on L channel)
  -> Segmentation            (MODNet matting -> YOLOv8-seg -> GrabCut)
  -> Dense Slice Analysis     (perpendicular cross-sections along the arm)
  -> Reliability Gate         (stability, arm-length consistency, lighting)
  -> Relaxed Scan            (10 stable frames captured)
  -> Flexed Scan             (10 stable frames captured)
  -> Robust Aggregation       (median after IQR outlier removal)
  -> Phase Consistency        (relaxed vs. flexed geometry guard)
  -> Scoring                 (weighted 1-10 Visual Biceps Score + estimated cm)
```

Module layout:

```text
FlexScan/
|-- app/
|   |-- camera.py            webcam open/read/release, error handling
|   |-- pose.py              YOLOv8-pose wrapper -> person selection and joints
|   |-- arm_analysis.py      arm polygon, matting/segmentation, illumination, slices, reliability gates
|   |-- segmentation.py      MODNet matting (ONNX) + YOLOv8-seg person mask (falls back to GrabCut)
|   |-- scoring.py           normalization, weighting, physical widths, rule-based advice
|   |-- state.py             scan state machine + stability/timeout logic
|   |-- ui.py                OpenCV overlay drawing + clickable buttons
|   |-- config.py            every tunable value lives here
|-- models/                  downloaded models (git-ignored)
|-- download_models.py       model download script (Ultralytics + Hugging Face)
|-- main.py                  application entry point
|-- tests/                   unit tests (no webcam required)
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

FlexScan downloads three models (no token or account needed):

- **yolov8n-pose.pt** - required, detects the upper-body keypoints
  (Ultralytics, AGPL-3.0)
- **yolov8n-seg.pt** - optional, provides a cleaner person mask for more
  reliable arm isolation. When absent, FlexScan falls back to OpenCV GrabCut.
  (Ultralytics, AGPL-3.0)
- **modnet_photographic.onnx** - optional, MODNet portrait matting
  (Apache-2.0, hosted on Hugging Face, `Xenova/modnet`). Gives a soft,
  lighting-stable arm boundary. When absent, FlexScan uses the seg model and
  then GrabCut. (No HF token is required.)

```bash
python download_models.py
```

Models are saved to `models/` and git-ignored. To change the filenames,
update `MODEL_FILENAME`, `SEG_MODEL_FILENAME` and `MATTING_MODEL_FILENAME`
in `app/config.py`.

## Running

```bash
python main.py
```

## How to use

```text
Press 's' (or click "Start Scan")
  -> Relax your arm, hold still until the relaxed frames are captured
  -> Press 'f' (or click "Continue")
  -> Flex your bicep, hold still until the flexed frames are captured
  -> Press 'a' (or click "Analyze")
  -> View your Visual Biceps Score and observations
  -> Press 's' again (or "Scan Again") to repeat, or 'q' to quit
```

Keyboard shortcuts: `s` start / scan again, `f` continue to flex, `a` analyze,
`q` quit.

If a phase cannot be captured reliably, FlexScan tells you exactly what is
wrong - lighting, arm visibility, movement - so you can reposition and try
again instead of chasing a meaningless number.

### Calibration (estimated physical widths)

At startup FlexScan asks for your **upper-arm length** (shoulder to elbow, in
cm). That single reference length lets the pixel widths be converted into
**estimated centimetres** (max / average per phase and the flex change). The
conversion assumes the arm is roughly parallel to the camera plane, so treat
the numbers as visual approximations - the UI says "ESTIMATED" and the result
panel repeats that they are visual estimates, not medical measurements.

You can skip the prompt (the default) to get relative-only scores, or set it
once instead of typing it each launch:

```bash
# .env
FLEXSCAN_UPPER_ARM_LENGTH_CM=35
```

## Landmark and measurement approach

### Pose landmarks and person selection

YOLOv8-pose returns the standard COCO-17 keypoints. When several people are in
frame, FlexScan picks the person whose shoulders, elbows, and wrists carry the
highest combined landmark confidence (weighted with the detection-box score),
rather than blindly trusting the largest box. Every upper-body landmark is
drawn on the feed, and the actively analyzed arm is highlighted with a thicker
colored skeleton. An arm is only used when its shoulder, elbow, and wrist all
clear the confidence threshold and the arm length is large enough to measure.

### Arm region polygon

Instead of a fixed rectangular crop, the upper-arm analysis region is a
**tapered polygon** derived from the shoulder-elbow geometry. The polygon is
built around the arm axis (shoulder to elbow) and tapers from a wider
mid-bicep belly toward the narrower shoulder and elbow ends, giving a more
anatomically accurate region. The polygon is shown in real time on the feed.

### Dense measurement slices and estimated widths

Inside the arm region, multiple **perpendicular cross-section lines** are
generated along the shoulder-elbow axis. Each slice measures the foreground
width of the arm at that position, and these are shown as orange measurement
lines with green edge dots during scanning. Because the slices are anchored to
the live shoulder-elbow geometry, they follow the actual arm outline even
when it moves slightly between frames. When calibration is enabled, the same
slice widths are converted to centimetres for the result panel.

### Distance and size normalization

Every raw width is divided by the shoulder-to-elbow arm length before it is
used or compared. Both a wider camera angle and a physically longer arm are
factored out of the numbers, so the same arm measured at different distances
produces consistent results. This normalization is applied per frame, which
also keeps small distance changes between the relaxed and flexed phases from
skewing the comparison.

## Robustness details

- **Person selection** - the tracked person maximizes box score times the mean
  confidence of the eight shoulder/elbow/wrist keypoints (`app/pose.py`).
- **Stability gate** - a frame is only used when the elbow and shoulder have
  stayed within a small pixel window and the arm length has not drifted by
  more than 15% over the last five frames (`app/state.py`).
- **Keypoint-jump rejection** - a stable frame whose arm length jumps well
  away from the phase median is skipped, because a mis-detected keypoint can
  otherwise distort one of the ten capture frames.
- **Outlier aggregation** - each component is aggregated with the median after
  discarding 1.5 x IQR outliers; width profiles are medians per slice position.
- **Lighting tolerance** - frames are illumination-normalized (CLAHE) first,
  edge thresholds are derived from the arm region's mean and standard
  deviation (`CANNY_EDGE_SIGMA`), and any frame whose arm pixels still average
  too dark or too bright is rejected as unreliable.
- **Segmentation robustness** - the arm is separated with MODNet portrait
  matting (soft, lighting-stable alpha), falling back to YOLOv8-seg and then
  GrabCut. The matting crop is anchored to the pose-derived upper-body box, so
  pose and segmentation work together.
- **Honest physical widths** - without a reference arm length no centimetre
  value is ever displayed; with one, every displayed number is marked
  "ESTIMATED" and the result panel states that it is a visual estimate, not a
  medical measurement.
- **Phase consistency guard** - if the arm length changed by more than 35%
  between the relaxed and flexed phases, the user moved relative to the
  camera, so the Flex Response component is marked unreliable rather than
  guessed.
- **Honest numbers** - no component is ever boosted, invented, or estimated.
  Flex Response is the median of the peak-slice change and the full-profile
  change, both real measurements.

## Scoring methodology

- **Peak Bulge** - the maximum cross-section width of the segmented arm,
  normalized by shoulder-to-elbow length (from the flexed phase)
- **Flex Response** - the median of (a) the change in peak bulge and (b) the
  median change across the whole width profile, both between flexed and
  relaxed and both normalized by arm length
- **Definition** - lighting-adaptive edge density inside the segmented arm
  region
- **Shape** - contour solidity (area vs. convex-hull area) of the flexed arm
- **Curvature** - mean absolute second derivative of the normalized width
  profile, capturing the bicep peak's sharpness

Each raw measurement is mapped onto a 1-10 scale using fixed, documented
ranges (`app/config.py: NORMALIZATION_RANGES`), then combined with these
weights into the overall score:

| Component       | Weight |
|-----------------|-------:|
| Peak Bulge      |    25% |
| Flex Response   |    30% |
| Definition      |    20% |
| Shape/Contour   |    15% |
| Curvature       |    10% |

If a component cannot be measured reliably (bad segmentation, changed camera
geometry, unusable lighting), it is excluded and the remaining weights are
renormalized. If too much of the scan is unreliable, FlexScan shows
**"Unable to obtain a reliable scan - adjust position/lighting and try
again."** instead of producing a misleading score.

**These weights and ranges are heuristic engineering choices for a visual
computer-vision demo - they are not medical or scientific standards.**

## Troubleshooting

| Problem | What to check |
|---|---|
| Camera not opening | Make sure no other app is using the webcam, and that the OS has granted camera permission. Try a different `FLEXSCAN_CAMERA_INDEX` if you have multiple cameras. |
| "Pose model not found" | Run `python download_models.py` before `python main.py`. |
| Matting model not used | FlexScan falls back automatically. Check that `models/modnet_photographic.onnx` exists; re-run `python download_models.py` to fetch it. |
| Wrong estimated widths | The reference arm length must match your real shoulder-to-elbow length, and you must stay roughly parallel to the camera. These are estimates, not measurements. |
| Download failure | Check your internet connection. If the model name has changed, update the filenames in `app/config.py`. |
| Poor lighting / low definition score | Use even, diffuse lighting facing your arm; avoid strong backlight or deep shadows. |
| Arm not detected | Make sure your shoulder, elbow, and wrist are all visible in frame, and step back so the whole upper arm is in view. |
| "Unable to obtain a reliable scan" | The captured frames failed the reliability gates. Even out the lighting, keep the arm fully in frame, hold still while the frames collect, and avoid leaning toward or away from the camera between the relaxed and flexed phases. |
| Low FPS | Close other apps using the camera or GPU; a smaller `yolov8n` model already runs on CPU, but a GPU will speed it up further. |

## Privacy

All webcam processing happens **locally** on your machine. FlexScan does
not upload video, images, or measurements anywhere - the only network
access it uses is the one-time model download in `download_models.py`.
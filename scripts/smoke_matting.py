import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app.arm_analysis import segment_arm_region, build_arm_region, aggregate_measurements, analyze_frame
from app.pose import ArmPose
from app.segmentation import get_matting_model

img = np.full((300, 300, 3), 200, dtype=np.uint8)
cv2.rectangle(img, (100, 40), (200, 220), (120, 120, 120), -1)

alpha = get_matting_model().matte_crop(img)
print("alpha:", None if alpha is None else (alpha.shape, float(alpha.min()), float(alpha.max())))

arm = ArmPose(shoulder=(100, 100), elbow=(200, 100), wrist=(210, 160), side="right", confidence=0.9)
region = build_arm_region(img, arm)
mask = segment_arm_region(img, region, person_box=(60, 20, 180, 240))
print("arm mask via matting:", None if mask is None else (mask.dtype, int(mask.sum() // 255)))

meas = analyze_frame(img, arm, reference_cm=30.0, person_box=(60, 20, 180, 240))
print("measurement reliable:", meas.reliable, "reason:", meas.reason)
if meas.reliable:
    print("peak_bulge:", round(meas.peak_bulge, 3), "widths_cm[0]:", round(meas.widths_cm[0], 2))
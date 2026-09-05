import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from app.camera import Camera
from app.pose import PoseDetector
from app.arm_analysis import analyze_frame, build_arm_region, segment_arm_region, measure_slice_widths, aggregate_measurements
from app.state import ScanSession, ScanState


def main():
    detector = PoseDetector()
    session = ScanSession()
    session.state = ScanState.RELAXED_SCAN
    session.arm_side = "right"

    with Camera() as cam:
        reasons = {}
        frames_captured = 0
        for i in range(80):
            frame = cam.read()
            pose = detector.detect(frame)
            arm = pose.arms.get("right") or pose.arms.get("left")
            if arm is None:
                continue
            if session.arm_side is None:
                session.arm_side = arm.side
            box = session._person_box(pose, frame.shape[:2])
            meas = analyze_frame(frame, arm, None, box)
            reason = meas.reason or "OK"
            reasons[reason] = reasons.get(reason, 0) + 1
            if meas.reliable:
                frames_captured += 1
                if frames_captured >= 10:
                    print("captured 10 reliable relaxed frames")
                    break

        print("reason counts:", reasons)
        print("arm_side:", session.arm_side, "reference_cm:", session.reference_cm)

        # diagnose one full unrolled pass on a fresh frame
        frame = cam.read()
        pose = detector.detect(frame)
        arm = pose.arms.get("right") or pose.arms.get("left")
        print("person_detected:", pose.person_detected, "arms:", list(pose.arms.keys()))
        if arm is None:
            print("no usable arm (conf/low length). confs:", pose.upper_body.confidences)
            return
        from app.arm_analysis import normalize_illumination
        work = normalize_illumination(frame)
        region = build_arm_region(work, arm)
        print("region:", "None" if region is None else (len(region.slices), round(region.arm_length, 1)))
        if region is None:
            return
        box = session._person_box(pose, frame.shape[:2])
        print("person_box:", box)
        mask = segment_arm_region(work, region, box)
        print("mask:", "None" if mask is None else (mask.dtype, mask.shape, int(mask.sum() // 255)))
        if mask is not None:
            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            roi = cv2.bitwise_and(gray, gray, mask=mask)
            print("roi lum min/max/mean:", int(roi[roi > 0].min()), int(roi[roi > 0].max()),
                  round(float(roi[roi > 0].mean()), 1))
            from app.arm_analysis import measure_definition, measure_shape, measure_curvature
            slices = measure_slice_widths(mask, region)
            print("slices:", None if not slices else len(slices), "nonempty:", sum(1 for s in slices if s.width_px > 0))
            if slices:
                widths = [s.width_px for s in slices]
                nonempty = [w for w in widths if w > 0]
                print("width px nonempty min/max/mean:", \
                    round(min(nonempty), 1), round(max(nonempty), 1), round(float(sum(nonempty) / len(nonempty)), 1))
                print("norm peak_bulge:", round(max(w / region.arm_length for w in nonempty), 3))
            print("definition:", measure_definition(work, mask), "shape:", measure_shape(mask),
                  "curvature:", measure_curvature(slices, region.arm_length))

        meas = analyze_frame(frame, arm, None, box)
        print("analyze_frame reliable:", meas.reliable, "reason:", meas.reason)
        print("arm_length:", meas.arm_length)


if __name__ == "__main__":
    main()
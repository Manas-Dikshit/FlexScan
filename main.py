"""
FlexScan entry point.

Run with:
    python main.py

Controls:
    - Press 's' to start scan or scan again
    - Press 'f' to continue to flex
    - Press 'a' to analyze
    - Press 'q' to quit
"""

from __future__ import annotations

import sys

import cv2

from app import config
from app.camera import Camera, CameraError
from app.pose import PoseDetector, ModelNotDownloadedError
from app.scoring import score_scan
from app.state import ScanSession, ScanState
from app.ui import (
    ButtonManager, Button, draw_pose_overlay, draw_arm_region,
    draw_measurement_points, draw_status, draw_result,
)


class FlexScanApp:
    def __init__(self):
        self.camera = Camera()
        self.pose_detector = PoseDetector()
        self.session = ScanSession()
        self.result = None
        self.buttons = ButtonManager()
        self._quit = False

    # -- button actions ---------------------------------------------------
    def _on_start(self):
        self.session.start_scan()

    def _on_continue_to_flex(self):
        self.session.proceed_to_flex_scan()

    def _on_analyze(self):
        self.session.proceed_to_analyze()
        if self.session.relaxed_aggregate and self.session.flexed_aggregate:
            self.result = score_scan(
                self.session.relaxed_aggregate,
                self.session.flexed_aggregate,
                self.session.reference_cm,
            )
        else:
            from app.scoring import ScanResult, ComponentScores
            self.result = ScanResult(
                overall_score=None,
                components=ComponentScores(None, None, None, None, None,
                                          unreliable=["peak_bulge", "flex_change", "definition", "shape", "curvature"]),
                advice=["Not enough reliable data was captured. Try again with better lighting."],
            )
        self.session.finish_analysis()

    def _on_scan_again(self):
        self.session.scan_again()
        self.result = None

    def _on_quit(self):
        self._quit = True

    def _setup_reference_length(self) -> None:
        """Ask for one reference measurement (upper-arm length in cm) if not set."""
        if self.session.reference_cm is not None:
            print(
                "[FlexScan] Calibration: "
                f"upper-arm length = {self.session.reference_cm:.1f} cm "
                "(set via FLEXSCAN_UPPER_ARM_LENGTH_CM or config)."
            )
            return
        print("[FlexScan] Optional calibration. Enter your upper-arm length")
        print("[FlexScan] (shoulder to elbow) in cm. Blank = relative score only.")
        try:
            raw = input("[FlexScan] Upper-arm length (cm): ").strip()
        except EOFError:
            raw = ""
        try:
            value = float(raw)
        except ValueError:
            value = 0.0
        if value > 0:
            self.session.reference_cm = value
            print(f"[FlexScan] Calibration set: {value:.1f} cm.")
        else:
            print("[FlexScan] No reference set - widths will not be estimated.")

    def _rebuild_buttons(self, frame_w: int, frame_h: int) -> None:
        buttons = []
        by = frame_h - 64
        primary = (90, 180, 110)   # highlighted primary action
        if self.session.state == ScanState.READY:
            buttons.append(Button("Start Scan  [s]", 16, by, 200, by + 44, self._on_start, primary))
        elif self.session.state == ScanState.RELAXED_CAPTURED:
            buttons.append(Button("Flex Now  [f]", 16, by, 200, by + 44, self._on_continue_to_flex, primary))
        elif self.session.state == ScanState.FLEX_CAPTURED:
            buttons.append(Button("Analyze  [a]", 16, by, 200, by + 44, self._on_analyze, primary))
        elif self.session.state == ScanState.RESULT:
            buttons.append(Button("Scan Again  [s]", 16, by, 200, by + 44, self._on_scan_again, primary))
        buttons.append(Button("Quit  [q]", frame_w - 146, by, frame_w - 16, by + 44, self._on_quit, (70, 70, 200)))
        self.buttons.set_buttons(buttons)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.buttons.handle_click(x, y)

    def run(self) -> int:
        self._setup_reference_length()

        try:
            self.camera.open()
        except CameraError as e:
            print(f"[FlexScan] Camera error: {e}")
            return 1

        try:
            self.pose_detector._ensure_loaded()
        except ModelNotDownloadedError as e:
            print(f"[FlexScan] {e}")
            self.camera.release()
            return 1

        cv2.namedWindow(config.WINDOW_NAME)
        cv2.setMouseCallback(config.WINDOW_NAME, self._mouse_callback)

        try:
            self._main_loop()
        finally:
            self.camera.release()
            cv2.destroyAllWindows()
        return 0

    def _main_loop(self) -> None:
        status_text = ""
        while not self._quit:
            try:
                frame = self.camera.read()
            except CameraError as e:
                print(f"[FlexScan] {e}")
                break

            pose = self.pose_detector.detect(frame)

            if self.session.state in (ScanState.RELAXED_SCAN, ScanState.FLEX_SCAN):
                status_text = self.session.update(frame, pose)

            # Draw overlays
            draw_pose_overlay(frame, pose, self.session.arm_side)

            if self.session.current_arm_region is not None:
                draw_arm_region(frame, self.session.current_arm_region)
                if self.session.current_arm_region.slices:
                    draw_measurement_points(
                        frame,
                        self.session.current_arm_region.slices,
                        self.session.current_arm_region.arm_length,
                    )

            draw_status(frame, self.session, status_text)
            if self.result is not None and self.session.state == ScanState.RESULT:
                draw_result(frame, self.result)

            h, w = frame.shape[:2]
            self._rebuild_buttons(w, h)
            self.buttons.draw(frame)

            disp = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            cv2.imshow(config.WINDOW_NAME, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                if self.session.state == ScanState.RESULT:
                    self._on_scan_again()
                else:
                    self._on_start()
            elif key == ord("f"):
                self._on_continue_to_flex()
            elif key == ord("a"):
                self._on_analyze()


def main() -> int:
    app = FlexScanApp()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())

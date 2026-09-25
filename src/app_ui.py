"""PyQt6 application for the Snapdragon Vision AI demo."""

from __future__ import annotations

import argparse
import sys

import cv2
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

from .model_loader import ModelConfig, ModelLoader
from .video_pipeline import VideoPipeline


class VisionWindow(QMainWindow):
    """Live preview and telemetry dashboard."""

    def __init__(self, pipeline: VideoPipeline) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.setWindowTitle("Snapdragon Vision AI")
        self.setMinimumSize(960, 640)

        self.preview = QLabel("Starting camera...")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(640, 480)
        self.telemetry = QLabel()
        self.telemetry.setStyleSheet("font-size: 16px; padding: 8px;")

        layout = QVBoxLayout()
        layout.addWidget(self.preview, stretch=1)
        layout.addWidget(self.telemetry)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(0)

    def _update_frame(self) -> None:
        try:
            result = self.pipeline.next_frame()
        except RuntimeError as error:
            self.timer.stop()
            self.telemetry.setText(f"Pipeline stopped: {error}")
            return

        rgb_frame = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        image = QImage(rgb_frame.data, width, height, channels * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pixmap)
        self.telemetry.setText(
            f"Backend: {result.backend}    |    Inference latency: {result.latency_ms:.1f} ms    |    "
            f"Live FPS: {result.fps:.1f}    |    Privacy regions: {len(result.detections)}"
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API name
        self.timer.stop()
        self.pipeline.close()
        super().closeEvent(event)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapdragon Vision AI webcam privacy demo")
    parser.add_argument("--model", required=True, help="Path to a quantized INT8 ONNX model")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--cpu", action="store_true", help="Skip QNN and prefer fallback providers")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    try:
        loader = ModelLoader(ModelConfig(args.model, prefer_qnn=not args.cpu))
        pipeline = VideoPipeline(loader, camera_index=args.camera)
        window = VisionWindow(pipeline)
        window.show()
        return app.exec()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Startup failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

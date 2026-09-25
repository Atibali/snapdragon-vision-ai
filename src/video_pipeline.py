"""Low-latency webcam capture, inference, privacy masking, and telemetry."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .model_loader import ModelLoader


@dataclass(frozen=True)
class Detection:
    """A normalized detection box and confidence score."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


@dataclass(frozen=True)
class FrameResult:
    """Processed frame and measurements for the UI."""

    frame: np.ndarray
    latency_ms: float
    fps: float
    detections: tuple[Detection, ...]
    backend: str


class VideoPipeline:
    """Capture and process frames using a reusable preallocated model path."""

    def __init__(
        self,
        model_loader: ModelLoader,
        camera_index: int = 0,
        input_size: tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.40,
    ) -> None:
        self.model_loader = model_loader
        self.capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Unable to open webcam index {camera_index}")

        self.input_width, self.input_height = input_size
        self.confidence_threshold = confidence_threshold
        self._last_frame_time = time.perf_counter()
        self._fps = 0.0

    def close(self) -> None:
        """Release the camera handle."""
        self.capture.release()

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize BGR camera data into the model's NCHW input tensor."""
        resized = cv2.resize(frame, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
        if "int8" in self.model_loader.input_type:
            return np.ascontiguousarray(tensor.astype(np.int8))
        return np.ascontiguousarray(tensor.astype(np.float32) / 255.0)

    def _decode_detections(self, outputs: Iterable[np.ndarray], frame_shape: tuple[int, ...]) -> tuple[Detection, ...]:
        """Decode common YOLO export output: [1, N, 6] or [N, 6]."""
        output = next(iter(outputs), np.empty((0, 6), dtype=np.float32))
        values = np.squeeze(output)
        if values.ndim != 2 or values.shape[1] < 5:
            return ()

        frame_height, frame_width = frame_shape[:2]
        detections: list[Detection] = []
        for row in values:
            score = float(row[4])
            if score < self.confidence_threshold:
                continue
            coordinates = row[:4].astype(np.float32)
            if np.max(coordinates) <= 1.5:
                coordinates[[0, 2]] *= frame_width
                coordinates[[1, 3]] *= frame_height
            x1, y1, x2, y2 = coordinates.astype(int).tolist()
            detections.append(Detection(max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2), score))
        return tuple(detections)

    @staticmethod
    def _apply_privacy_mask(frame: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
        """Blur detected regions locally, preserving the rest of the frame."""
        processed = frame.copy()
        for detection in detections:
            region = processed[detection.y1:detection.y2, detection.x1:detection.x2]
            if region.size == 0:
                continue
            processed[detection.y1:detection.y2, detection.x1:detection.x2] = cv2.GaussianBlur(region, (0, 0), sigmaX=18)
        return processed

    def next_frame(self) -> FrameResult:
        """Capture, infer, mask, and return one frame."""
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("Webcam frame capture failed")

        start = time.perf_counter()
        outputs = self.model_loader.infer(self._preprocess(frame))
        detections = self._decode_detections(outputs, frame.shape)
        processed = self._apply_privacy_mask(frame, detections)
        latency_ms = (time.perf_counter() - start) * 1000.0

        now = time.perf_counter()
        interval = now - self._last_frame_time
        self._last_frame_time = now
        if interval > 0:
            instantaneous_fps = 1.0 / interval
            self._fps = instantaneous_fps if self._fps == 0 else (self._fps * 0.85) + (instantaneous_fps * 0.15)

        return FrameResult(processed, latency_ms, self._fps, detections, self.model_loader.execution_backend)

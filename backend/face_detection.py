from dataclasses import dataclass

import cv2
import numpy as np

from utils.config import get_config
from utils.logger import get_logger

log = get_logger(__name__)

# Haar cascade for fast CPU-based detection (fallback / lightweight option)
_HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


@dataclass
class DetectedFace:
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def as_crop(self, frame: np.ndarray, padding: int = 20) -> np.ndarray:
        h_img, w_img = frame.shape[:2]
        x1 = max(0, self.x - padding)
        y1 = max(0, self.y - padding)
        x2 = min(w_img, self.x + self.w + padding)
        y2 = min(h_img, self.y + self.h + padding)
        return frame[y1:y2, x1:x2]


class FaceDetector:
    """
    Multi-backend face detector.
    Primary: OpenCV DNN (faster, more accurate than Haar).
    Fallback: Haar cascade.
    """

    def __init__(self):
        self._cfg = get_config()
        self._haar = cv2.CascadeClassifier(_HAAR_CASCADE_PATH)
        self._dnn_net = self._load_dnn_model()
        log.info("FaceDetector initialised (backend={})", self._cfg.detection_backend)

    def _load_dnn_model(self):
        """Load OpenCV DNN face detector (pre-trained Caffe model)."""
        try:
            import urllib.request
            from pathlib import Path

            weights_dir = self._cfg.embeddings_dir.parent / "weights"
            weights_dir.mkdir(exist_ok=True)
            prototxt = weights_dir / "deploy.prototxt"
            caffemodel = weights_dir / "res10_300x300_ssd_iter_140000.caffemodel"

            if prototxt.exists() and caffemodel.exists():
                net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
                log.info("DNN face detector loaded from cache")
                return net
        except Exception as e:
            log.warning("DNN model not available ({}), falling back to Haar cascade", e)
        return None

    def detect(self, frame: np.ndarray, min_confidence: float = 0.5) -> list[DetectedFace]:
        """Detect all faces in a BGR frame. Returns list of DetectedFace."""
        if self._dnn_net is not None:
            return self._detect_dnn(frame, min_confidence)
        return self._detect_haar(frame)

    def _detect_dnn(self, frame: np.ndarray, min_confidence: float) -> list[DetectedFace]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self._dnn_net.setInput(blob)
        detections = self._dnn_net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < min_confidence:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            faces.append(DetectedFace(x=x1, y=y1, w=x2 - x1, h=y2 - y1, confidence=confidence))
        log.debug("DNN detected {} face(s)", len(faces))
        return faces[: self._cfg.max_faces_per_frame]

    def _detect_haar(self, frame: np.ndarray) -> list[DetectedFace]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        faces = [DetectedFace(x=int(x), y=int(y), w=int(w), h=int(h)) for (x, y, w, h) in rects]
        log.debug("Haar detected {} face(s)", len(faces))
        return faces[: self._cfg.max_faces_per_frame]

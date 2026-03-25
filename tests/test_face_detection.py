import numpy as np
import pytest

from backend.face_detection import FaceDetector, DetectedFace


@pytest.fixture
def detector():
    return FaceDetector()


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def white_frame():
    return np.ones((480, 640, 3), dtype=np.uint8) * 255


def test_detect_returns_list(detector, blank_frame):
    result = detector.detect(blank_frame)
    assert isinstance(result, list)


def test_no_faces_in_blank_frame(detector, blank_frame):
    result = detector.detect(blank_frame)
    assert len(result) == 0


def test_detected_face_bbox(detector):
    face = DetectedFace(x=10, y=20, w=80, h=90)
    assert face.bbox == (10, 20, 80, 90)


def test_face_crop_within_bounds():
    face = DetectedFace(x=100, y=100, w=100, h=100)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    crop = face.as_crop(frame, padding=10)
    assert crop.shape[0] > 0
    assert crop.shape[1] > 0


def test_max_faces_limit(detector, blank_frame):
    result = detector.detect(blank_frame)
    cfg_max = detector._cfg.max_faces_per_frame
    assert len(result) <= cfg_max

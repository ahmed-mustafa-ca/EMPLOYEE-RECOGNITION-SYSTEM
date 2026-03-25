import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.attendance_manager import AttendanceManager
from backend.face_detection import FaceDetector, DetectedFace
from backend.face_recognition import FaceRecognizer, RecognitionResult
from database.db_handler import DBHandler


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = DBHandler(db_path=Path(tmpdir) / "test.db")
        db.add_employee("EMP001", "Alice", "Engineering", "alice@test.com")
        yield db


@pytest.fixture
def mock_detector():
    detector = MagicMock(spec=FaceDetector)
    face = DetectedFace(x=50, y=50, w=100, h=100)
    face.as_crop = MagicMock(return_value=np.zeros((160, 160, 3), dtype=np.uint8))
    detector.detect.return_value = [face]
    return detector


@pytest.fixture
def mock_recognizer_matched():
    rec = MagicMock(spec=FaceRecognizer)
    rec.recognize.return_value = RecognitionResult(
        employee_id="EMP001", name="Alice", confidence=0.95, matched=True
    )
    return rec


@pytest.fixture
def mock_recognizer_unknown():
    rec = MagicMock(spec=FaceRecognizer)
    rec.recognize.return_value = RecognitionResult(
        employee_id="", name="Unknown", confidence=0.1, matched=False
    )
    return rec


def test_process_frame_marks_attendance(db, mock_detector, mock_recognizer_matched):
    mgr = AttendanceManager(db, mock_detector, mock_recognizer_matched)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, events = mgr.process_frame(frame)
    assert len(events) == 1
    assert events[0]["employee_id"] == "EMP001"


def test_process_frame_unknown_no_event(db, mock_detector, mock_recognizer_unknown):
    mgr = AttendanceManager(db, mock_detector, mock_recognizer_unknown)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, events = mgr.process_frame(frame)
    assert len(events) == 0


def test_cooldown_prevents_duplicate_marking(db, mock_detector, mock_recognizer_matched):
    mgr = AttendanceManager(db, mock_detector, mock_recognizer_matched)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, events1 = mgr.process_frame(frame)
    _, events2 = mgr.process_frame(frame)   # Within cooldown window
    assert len(events1) == 1
    assert len(events2) == 0


def test_get_today_attendance(db, mock_detector, mock_recognizer_matched):
    mgr = AttendanceManager(db, mock_detector, mock_recognizer_matched)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mgr.process_frame(frame)
    records = mgr.get_today_attendance()
    assert len(records) >= 1

from datetime import datetime, date, timedelta

import numpy as np

from backend.face_detection import FaceDetector, DetectedFace
from backend.face_recognition import FaceRecognizer, RecognitionResult
from database.db_handler import DBHandler
from utils.config import get_config
from utils.image_utils import draw_face_box
from utils.logger import get_logger

log = get_logger(__name__)


class AttendanceManager:
    """
    Orchestrates real-time attendance marking from live video frames.
    - Detects all faces in a frame
    - Recognises each face against stored embeddings
    - Marks attendance in the DB with cooldown to prevent duplicates
    """

    def __init__(self, db: DBHandler, detector: FaceDetector, recognizer: FaceRecognizer):
        self._db = db
        self._detector = detector
        self._recognizer = recognizer
        self._cfg = get_config()
        # In-memory cooldown: {employee_id: last_marked_datetime}
        self._cooldown: dict[str, datetime] = {}

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Process a single video frame.
        Returns:
            annotated_frame: frame with bounding boxes drawn
            events: list of attendance events that occurred this frame
        """
        annotated = frame.copy()
        events = []
        faces: list[DetectedFace] = self._detector.detect(frame)

        for face in faces:
            crop = face.as_crop(frame)
            result: RecognitionResult = self._recognizer.recognize(crop)

            if result.matched:
                color = (0, 200, 0)          # Green
                label = f"{result.name} | Employee"
                event = self._try_mark_attendance(result)
                if event:
                    events.append(event)
            else:
                color = (0, 0, 220)          # Red
                label = "Not Employee"

            annotated = draw_face_box(
                annotated, face.x, face.y, face.w, face.h,
                label=label, confidence=result.confidence, color=color
            )

        return annotated, events

    def _try_mark_attendance(self, result: RecognitionResult) -> dict | None:
        """Mark attendance if outside cooldown window."""
        now = datetime.now()
        last = self._cooldown.get(result.employee_id)
        if last and (now - last).total_seconds() < self._cfg.attendance_cooldown:
            return None  # Still in cooldown

        self._cooldown[result.employee_id] = now
        record = self._db.mark_attendance(result.employee_id, result.name, result.confidence)
        log.info("Attendance marked: {} ({}) @ {}", result.name, result.employee_id, now.strftime("%H:%M:%S"))
        return {"employee_id": result.employee_id, "name": result.name, "time": now, "record": record}

    def get_today_attendance(self) -> list[dict]:
        return self._db.get_attendance_by_date(date.today())

    def get_attendance_report(self, from_date: date, to_date: date) -> list[dict]:
        return self._db.get_attendance_range(from_date, to_date)

    def get_monthly_summary(self, year: int, month: int) -> list[dict]:
        from_date = date(year, month, 1)
        # Last day of month
        if month == 12:
            to_date = date(year + 1, 1, 1) - timedelta(days=1)
        else:
            to_date = date(year, month + 1, 1) - timedelta(days=1)
        return self._db.get_attendance_range(from_date, to_date)

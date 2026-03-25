# Lazy imports — TensorFlow and DeepFace only load when explicitly used,
# not on every `import backend` call. This keeps startup fast.

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.face_detection import FaceDetector
    from backend.face_recognition import FaceRecognizer
    from backend.embedding_manager import EmbeddingManager
    from backend.employee_manager import EmployeeManager
    from backend.attendance_manager import AttendanceManager
    from backend.pipeline import FaceRecognitionPipeline
    from backend.registration import register_employee

__all__ = [
    "FaceDetector",
    "FaceRecognizer",
    "EmbeddingManager",
    "EmployeeManager",
    "AttendanceManager",
    "FaceRecognitionPipeline",
    "register_employee",
]

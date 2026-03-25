import numpy as np

from backend.embedding_manager import EmbeddingManager
from backend.face_detection import FaceDetector
from backend.face_recognition import FaceRecognizer
from database.db_handler import DBHandler
from utils.config import get_config
from utils.image_utils import save_image
from utils.logger import get_logger
from utils.validators import validate_employee_id, validate_name, validate_email

log = get_logger(__name__)


class EmployeeManager:
    """
    High-level API for registering, updating, and removing employees.
    Coordinates between the database, image storage, and embeddings.
    """

    def __init__(
        self,
        db: DBHandler,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        emb_mgr: EmbeddingManager,
    ):
        self._db = db
        self._detector = detector
        self._recognizer = recognizer
        self._emb_mgr = emb_mgr
        self._cfg = get_config()

    def register(
        self,
        employee_id: str,
        name: str,
        department: str,
        email: str,
        images: list[np.ndarray],
    ) -> dict:
        """
        Register a new employee.
        - Validates inputs
        - Detects and crops faces from provided images
        - Generates embeddings and stores them
        - Saves employee record to the database
        Returns a status dict.
        """
        # --- Validation ---
        if not validate_employee_id(employee_id):
            return {"success": False, "message": "Invalid employee ID format."}
        if not validate_name(name):
            return {"success": False, "message": "Invalid name format."}
        if not validate_email(email):
            return {"success": False, "message": "Invalid email address."}
        if self._db.employee_exists(employee_id):
            return {"success": False, "message": f"Employee ID '{employee_id}' already registered."}

        # --- Face detection & embedding extraction ---
        embeddings = []
        saved_images = []
        emp_img_dir = self._cfg.employee_images_dir / employee_id
        emp_img_dir.mkdir(parents=True, exist_ok=True)

        for idx, img in enumerate(images):
            faces = self._detector.detect(img)
            if not faces:
                log.warning("No face detected in image {} for employee '{}'", idx, employee_id)
                continue
            # Use the largest detected face
            face = max(faces, key=lambda f: f.w * f.h)
            crop = face.as_crop(img)
            emb = self._recognizer.get_embedding(crop)
            if emb is None:
                continue
            embeddings.append(emb)
            img_path = emp_img_dir / f"{idx:03d}.jpg"
            save_image(crop, img_path)
            saved_images.append(str(img_path))

        if not embeddings:
            return {"success": False, "message": "Could not extract face embeddings from provided images."}

        # --- Persist ---
        self._emb_mgr.save(employee_id, name, embeddings)
        self._db.add_employee(
            employee_id, name, department, email,
            image_path=saved_images[0] if saved_images else None,
            embedding=embeddings[0] if embeddings else None,
        )
        log.info("Registered employee '{}' with {} embedding(s)", employee_id, len(embeddings))
        return {
            "success": True,
            "message": f"Employee '{name}' registered successfully.",
            "embeddings_count": len(embeddings),
        }

    def update_employee(self, employee_id: str, **fields) -> bool:
        """Update mutable fields (name, department, email) in the database."""
        if not self._db.employee_exists(employee_id):
            return False
        self._db.update_employee(employee_id, **fields)
        return True

    def delete_employee(self, employee_id: str) -> bool:
        if not self._db.employee_exists(employee_id):
            return False
        self._db.delete_employee(employee_id)
        self._emb_mgr.delete(employee_id)
        # Remove images
        import shutil
        img_dir = self._cfg.employee_images_dir / employee_id
        if img_dir.exists():
            shutil.rmtree(img_dir)
        log.info("Deleted employee '{}'", employee_id)
        return True

    def get_all_employees(self) -> list[dict]:
        return self._db.get_all_employees()

    def get_employee(self, employee_id: str) -> dict | None:
        return self._db.get_employee(employee_id)

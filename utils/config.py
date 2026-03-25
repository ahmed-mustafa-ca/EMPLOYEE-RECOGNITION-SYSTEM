import os
from pathlib import Path
from functools import lru_cache

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent


class Config:
    """Central configuration loader. Merges settings.yaml with .env overrides."""

    def __init__(self, config_path: str = "config/settings.yaml"):
        self._path = ROOT_DIR / config_path
        self._data: dict = self._load()

    def _load(self) -> dict:
        with open(self._path, "r") as f:
            return yaml.safe_load(f)

    # --- Recognition ---
    @property
    def recognition_model(self) -> str:
        return os.getenv("RECOGNITION_MODEL", self._data["recognition"]["model"])

    @property
    def detection_backend(self) -> str:
        return os.getenv("DETECTION_BACKEND", self._data["recognition"]["detection_backend"])

    @property
    def similarity_threshold(self) -> float:
        return float(os.getenv("SIMILARITY_THRESHOLD", self._data["recognition"]["similarity_threshold"]))

    @property
    def distance_metric(self) -> str:
        return self._data["recognition"]["distance_metric"]

    @property
    def max_faces_per_frame(self) -> int:
        return int(os.getenv("MAX_FACES_PER_FRAME", self._data["recognition"]["max_faces_per_frame"]))

    # --- Attendance ---
    @property
    def attendance_cooldown(self) -> int:
        return int(os.getenv("ATTENDANCE_COOLDOWN_SECONDS", self._data["attendance"]["cooldown_seconds"]))

    # --- Database ---
    @property
    def db_path(self) -> Path:
        raw = os.getenv("DB_PATH", self._data["database"]["path"])
        return ROOT_DIR / raw

    # --- Paths ---
    @property
    def employee_images_dir(self) -> Path:
        raw = os.getenv("EMPLOYEE_IMAGES_DIR", self._data["paths"]["employee_images"])
        path = ROOT_DIR / raw
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def embeddings_dir(self) -> Path:
        raw = os.getenv("EMBEDDINGS_DIR", self._data["paths"]["embeddings"])
        path = ROOT_DIR / raw
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def temp_dir(self) -> Path:
        raw = os.getenv("TEMP_DIR", self._data["paths"]["temp"])
        path = ROOT_DIR / raw
        path.mkdir(parents=True, exist_ok=True)
        return path

    # --- Webcam ---
    @property
    def webcam_device(self) -> int:
        return self._data["webcam"]["device_index"]

    @property
    def webcam_resolution(self) -> tuple[int, int]:
        return self._data["webcam"]["frame_width"], self._data["webcam"]["frame_height"]


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config()
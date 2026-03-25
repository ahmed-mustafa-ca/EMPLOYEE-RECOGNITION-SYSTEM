from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from deepface import DeepFace

from backend.embedding_manager import EmbeddingManager
from utils.config import get_config
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RecognitionResult:
    employee_id: str
    name: str
    confidence: float                               # cosine similarity 0.0–1.0
    matched: bool
    bbox: tuple[int, int, int, int] | None = None  # (x, y, w, h) pixel coords


def _unknown(bbox: tuple[int, int, int, int] | None = None) -> "RecognitionResult":
    return RecognitionResult(employee_id="", name="Unknown", confidence=0.0, matched=False, bbox=bbox)


UNKNOWN = _unknown()


class FaceRecognizer:
    """
    Compares a face crop against stored embeddings using DeepFace.
    Uses cosine similarity for fast in-memory nearest-neighbour search.

    Performance
    ───────────
    Embeddings are loaded from disk once and cached in memory.
    Call reload_embeddings() after registering or deleting an employee
    to refresh the cache.
    """

    def __init__(self, embedding_manager: EmbeddingManager):
        self._cfg = get_config()
        self._emb_mgr = embedding_manager
        # In-memory cache: list of (employee_id, name, embedding)
        self._cache: list[tuple[str, str, np.ndarray]] | None = None
        log.info(
            "FaceRecognizer ready (model={}, metric={})",
            self._cfg.recognition_model,
            self._cfg.distance_metric,
        )

    def reload_embeddings(self) -> None:
        """Invalidate the in-memory cache so the next recognize() reloads from disk."""
        self._cache = None
        log.info("Embedding cache cleared — will reload on next recognition")

    def _get_embeddings(self) -> list[tuple[str, str, np.ndarray]]:
        """Return cached embeddings, loading from disk on first access."""
        if self._cache is None:
            self._cache = self._emb_mgr.load_all()
            log.debug("Loaded {} embedding vector(s) into cache", len(self._cache))
        return self._cache

    def get_embedding(self, face_crop: np.ndarray) -> np.ndarray | None:
        """Extract a 512-D embedding vector from a face crop (BGR numpy array)."""
        try:
            result = DeepFace.represent(
                img_path=face_crop,
                model_name=self._cfg.recognition_model,
                detector_backend="skip",   # face already cropped
                enforce_detection=False,
            )
            return np.array(result[0]["embedding"], dtype=np.float32)
        except Exception as e:
            log.warning("Embedding extraction failed: {}", e)
            return None

    def recognize(
        self,
        face_crop: np.ndarray,
        bbox: tuple[int, int, int, int] | None = None,
    ) -> RecognitionResult:
        """
        Match a face crop against all stored embeddings.

        Parameters
        ----------
        face_crop : BGR numpy array of the cropped face region.
        bbox      : (x, y, w, h) of this face in the original frame.
                    Passed through to the returned RecognitionResult.

        Returns
        -------
        RecognitionResult with .name, .confidence, .matched, and .bbox.
        """
        query_emb = self.get_embedding(face_crop)
        if query_emb is None:
            return _unknown(bbox)

        all_embeddings = self._get_embeddings()
        if not all_embeddings:
            log.debug("No embeddings stored yet")
            return _unknown(bbox)

        best_id, best_name, best_sim = None, None, -1.0
        for emp_id, name, stored_emb in all_embeddings:
            sim = self._cosine_similarity(query_emb, stored_emb)
            if sim > best_sim:
                best_sim, best_id, best_name = sim, emp_id, name

        # similarity_threshold is a cosine similarity value (higher = stricter).
        # A face is matched only if its cosine similarity >= threshold.
        threshold = self._cfg.similarity_threshold
        matched = best_sim >= threshold
        log.debug(
            "Best match: {} ({:.3f}) threshold={:.3f} matched={}",
            best_id, best_sim, threshold, matched,
        )
        return RecognitionResult(
            employee_id=best_id or "",
            name=best_name if matched else "Unknown",
            confidence=float(best_sim),
            matched=matched,
            bbox=bbox,
        )

    def recognize_batch(
        self,
        face_crops: list[np.ndarray],
        bboxes: list[tuple[int, int, int, int]] | None = None,
    ) -> list[RecognitionResult]:
        """Recognize multiple face crops. Optionally pass matching bboxes list."""
        if bboxes is None:
            bboxes = [None] * len(face_crops)
        return [self.recognize(crop, bbox) for crop, bbox in zip(face_crops, bboxes)]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

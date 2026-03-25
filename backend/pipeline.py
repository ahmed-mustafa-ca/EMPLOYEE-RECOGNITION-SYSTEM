"""
pipeline.py — Unified real-time face recognition pipeline.

Flow per frame
──────────────
1. FaceDetector  → detects all faces (OpenCV DNN, Haar cascade fallback)
2. FaceRecognizer → for each detected face:
      a. Extract 512-D Facenet512 embedding via DeepFace
      b. Cosine similarity search against cached employee embeddings
      c. Classify as Employee (≥ threshold) or Unknown
3. Return list[RecognitionResult] — one entry per face, each containing:
      .name           str   — "John Doe" or "Unknown"
      .confidence     float — cosine similarity score (0.0–1.0)
      .matched        bool  — True if above similarity threshold
      .employee_id    str   — "" for unknown faces
      .bbox           tuple — (x, y, w, h) in pixel coordinates

Performance optimisations
─────────────────────────
• Embeddings loaded from disk once and cached in RAM.
  Call pipeline.reload_embeddings() after register/delete operations.
• Frame skipping: full detect+recognise runs every (skip_frames + 1) frames.
  Intermediate frames reuse last results — keeps UI smooth at 30 fps while
  recognition only runs at ~10–15 fps.
• Minimum face size filter rejects noise before the expensive embedding call.
"""
from __future__ import annotations

import numpy as np

from backend.embedding_manager import EmbeddingManager
from backend.face_detection import FaceDetector, DetectedFace
from backend.face_recognition import FaceRecognizer, RecognitionResult, _unknown
from utils.config import get_config
from utils.logger import get_logger

log = get_logger(__name__)


class FaceRecognitionPipeline:
    """
    End-to-end real-time face recognition pipeline.

    Parameters
    ----------
    skip_frames : int
        Number of frames to skip between full detection+recognition runs.
        0 = process every frame (slowest, most accurate).
        2 = process 1 in 3 frames (default, good balance for 30 fps webcam).
    min_face_size : int | None
        Minimum face width AND height in pixels. Faces smaller than this
        are ignored. Defaults to config value (settings.yaml).
    """

    def __init__(self, skip_frames: int = 2, min_face_size: int | None = None):
        self._cfg = get_config()
        self._emb_mgr = EmbeddingManager()
        self._detector = FaceDetector()
        self._recognizer = FaceRecognizer(self._emb_mgr)
        self._skip_frames = max(0, skip_frames)
        self._min_face_size = min_face_size or self._cfg._data["recognition"]["min_face_size"]
        self._frame_count: int = 0
        self._last_results: list[RecognitionResult] = []
        log.info(
            "FaceRecognitionPipeline ready (skip_frames={}, min_face_px={})",
            self._skip_frames, self._min_face_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> list[RecognitionResult]:
        """
        Process one BGR video frame.

        Returns a list of RecognitionResult — one per detected face.
        Each result exposes:
            .name           "John Doe" | "Unknown"
            .confidence     cosine similarity (0.0–1.0)
            .matched        True when above threshold
            .employee_id    "" for unknown faces
            .bbox           (x, y, w, h) pixel coordinates in the original frame

        On skipped frames the previous results are returned unchanged,
        so callers can always draw fresh bounding boxes every frame.
        """
        self._frame_count += 1

        # Return cached results on skipped frames (still valid for drawing)
        if self._skip_frames > 0 and self._frame_count % (self._skip_frames + 1) != 0:
            return self._last_results

        faces: list[DetectedFace] = self._detector.detect(frame)

        # Filter out faces that are too small to embed reliably
        faces = [
            f for f in faces
            if f.w >= self._min_face_size and f.h >= self._min_face_size
        ]

        if not faces:
            self._last_results = []
            return []

        results: list[RecognitionResult] = []
        for face in faces:
            crop = face.as_crop(frame)
            if crop.size == 0:
                results.append(_unknown(bbox=(face.x, face.y, face.w, face.h)))
                continue
            result = self._recognizer.recognize(
                crop,
                bbox=(face.x, face.y, face.w, face.h),
            )
            results.append(result)

        log.debug(
            "Frame {}: {} face(s) detected, {} matched",
            self._frame_count,
            len(faces),
            sum(r.matched for r in results),
        )
        self._last_results = results
        return results

    def reload_embeddings(self) -> None:
        """
        Invalidate the in-memory embedding cache.
        Call this after registering a new employee or deleting one so the
        pipeline picks up the latest embeddings on the next frame.
        """
        self._recognizer.reload_embeddings()

    def reset(self) -> None:
        """Reset frame counter and clear cached results (e.g. on stream restart)."""
        self._frame_count = 0
        self._last_results = []
        log.debug("Pipeline reset")

    # ------------------------------------------------------------------
    # Properties (read-only)
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def last_results(self) -> list[RecognitionResult]:
        """The most recent recognition results (may be from a skipped frame)."""
        return self._last_results

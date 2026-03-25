"""
registration.py — Standalone employee registration function.

Handles the full pipeline:
  name + images  →  face crop  →  DeepFace embedding  →  SQLite + .pkl

Duplicate prevention (two layers)
──────────────────────────────────
1. employee_id uniqueness  : checked in DB before any processing.
2. Name-based near-duplicate check : warns if a similarly-named employee
   already exists (e.g. "John Doe" vs "john doe"), but does not block —
   the caller decides.

Multiple images
───────────────
Pass a list with as many photos as you have (different angles, lighting).
All successfully embedded images are stored.  The DB holds one representative
embedding (first); the .pkl file holds all of them for richer matching.

Usage
─────
    from backend.registration import register_employee
    import cv2

    images = [cv2.imread("john_front.jpg"), cv2.imread("john_side.jpg")]
    result = register_employee(name="John Doe", images=images)
    # result → {"success": True, "employee_id": "EMP_JohnDoe_a3f1",
    #            "embeddings_saved": 2, "images_saved": 2}
"""
from __future__ import annotations

import pickle
import re
import uuid
from pathlib import Path

import cv2
import numpy as np
from deepface import DeepFace

from database.db_handler import DBHandler, get_db
from backend.embedding_manager import EmbeddingManager
from utils.config import get_config
from utils.image_utils import save_image
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def register_employee(
    name: str,
    images: list[np.ndarray],
    *,
    employee_id: str | None = None,
    department: str = "",
    email: str = "",
    db: DBHandler | None = None,
    model_name: str = "Facenet512",
    min_face_confidence: float = 0.5,
) -> dict:
    """
    Register an employee: extract face embeddings and persist everything.

    Parameters
    ----------
    name : str
        Full display name, e.g. "Jane Smith".
    images : list[np.ndarray]
        One or more BGR face images (numpy arrays from cv2 or webcam frames).
        Each image is processed independently; usable embeddings are kept.
        Pass pre-cropped face crops for best results, or full frames — the
        function will auto-detect and crop the largest face.
    employee_id : str | None
        Unique business ID (e.g. "EMP001").  Auto-generated from name + short
        UUID suffix if omitted.
    department : str
        Optional department name.
    email : str
        Optional contact email.  Must be unique in the DB if provided.
    db : DBHandler | None
        Injected DB instance.  Uses the global singleton if None.
    model_name : str
        DeepFace model for embedding.  Default: "Facenet512" (512-D, accurate).
    min_face_confidence : float
        Minimum confidence for OpenCV DNN face detection (0–1).
        Ignored if the image is already a tight face crop.

    Returns
    -------
    dict with keys:
        success          bool
        message          str   — human-readable status
        employee_id      str   — the ID used (auto or provided)
        embeddings_saved int   — number of embeddings successfully stored
        images_saved     int   — number of images saved to disk
        skipped          int   — images where no face / embedding was found
    """
    # ── 1. Validate inputs ────────────────────────────────────────────────
    name = name.strip()
    if not name or len(name) < 2:
        return _fail("Name must be at least 2 characters.")

    cfg = get_config()
    db = db or get_db()
    emb_mgr = EmbeddingManager()

    if not images:
        return _fail("At least one image is required.")

    # ── 2. Generate or validate employee_id ───────────────────────────────
    if employee_id is None:
        employee_id = _generate_id(name)
    else:
        employee_id = employee_id.strip()

    if db.employee_exists(employee_id):
        # Employee already exists — append new images instead of re-registering
        log.info("Employee '{}' exists — appending new images", employee_id)
        return _append_images(
            employee_id=employee_id,
            name=name,
            images=images,
            db=db,
            emb_mgr=emb_mgr,
            cfg=cfg,
            model_name=model_name,
            min_face_confidence=min_face_confidence,
        )

    # ── 3. Near-duplicate name check ──────────────────────────────────────
    existing = db.get_all_employees(active_only=False)
    for emp in existing:
        if emp["name"].lower() == name.lower():
            log.warning(
                "Near-duplicate name detected: '{}' already registered as '{}'",
                name, emp["employee_id"],
            )
            # Not blocking — caller can decide. Logged as a warning.

    # ── 4. Extract embeddings from each image ─────────────────────────────
    embeddings: list[np.ndarray] = []
    saved_paths: list[str] = []
    skipped = 0

    emp_img_dir = cfg.employee_images_dir / employee_id
    emp_img_dir.mkdir(parents=True, exist_ok=True)

    for idx, img in enumerate(images):
        if img is None or img.size == 0:
            log.warning("Image {} is empty — skipping", idx)
            skipped += 1
            continue

        # Auto-detect and crop the largest face (handles full-frame images)
        face_crop = _crop_largest_face(img, min_confidence=min_face_confidence)
        if face_crop is None:
            log.warning("No face detected in image {} — skipping", idx)
            skipped += 1
            continue

        emb = _extract_embedding(face_crop, model_name)
        if emb is None:
            log.warning("Embedding extraction failed for image {} — skipping", idx)
            skipped += 1
            continue

        embeddings.append(emb)
        img_path = emp_img_dir / f"{idx:03d}.jpg"
        save_image(face_crop, img_path)
        saved_paths.append(str(img_path))

    if not embeddings:
        return _fail(
            f"Could not extract any face embeddings from {len(images)} image(s). "
            "Ensure images contain a clearly visible face.",
            employee_id=employee_id,
        )

    # ── 5. Persist embeddings (.pkl) ──────────────────────────────────────
    emb_mgr.save(employee_id, name, embeddings)

    # ── 6. Persist employee record (SQLite) ───────────────────────────────
    db.add_employee(
        employee_id=employee_id,
        name=name,
        department=department,
        email=email or f"{_slug(name)}@placeholder.local",
        image_path=saved_paths[0],          # representative face image
        embedding=embeddings[0],            # first embedding for quick DB lookup
    )

    log.info(
        "Registered '{}' ({}) — {} embedding(s), {} image(s) saved",
        name, employee_id, len(embeddings), len(saved_paths),
    )
    return {
        "success": True,
        "message": f"'{name}' registered successfully.",
        "employee_id": employee_id,
        "embeddings_saved": len(embeddings),
        "images_saved": len(saved_paths),
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _append_images(
    employee_id: str,
    name: str,
    images: list[np.ndarray],
    db: DBHandler,
    emb_mgr: EmbeddingManager,
    cfg,
    model_name: str,
    min_face_confidence: float,
) -> dict:
    """Add more face images to an already-registered employee."""
    emp_img_dir = cfg.employee_images_dir / employee_id
    emp_img_dir.mkdir(parents=True, exist_ok=True)

    # Count existing images so filenames don't collide
    existing_count = len(list(emp_img_dir.glob("*.jpg")))

    new_embeddings: list[np.ndarray] = []
    skipped = 0

    for i, img in enumerate(images):
        if img is None or img.size == 0:
            skipped += 1
            continue

        face_crop = _crop_largest_face(img, min_confidence=min_face_confidence)
        if face_crop is None:
            skipped += 1
            continue

        emb = _extract_embedding(face_crop, model_name)
        if emb is None:
            skipped += 1
            continue

        new_embeddings.append(emb)
        img_path = emp_img_dir / f"{existing_count + i:03d}.jpg"
        save_image(face_crop, img_path)

    if not new_embeddings:
        return _fail(
            "No new face embeddings could be extracted from the provided images.",
            employee_id=employee_id,
        )

    # Append new embeddings to the existing .pkl (preserves old embeddings)
    for emb in new_embeddings:
        emb_mgr.append(employee_id, name, emb)

    log.info(
        "Appended {} new embedding(s) for employee '{}'",
        len(new_embeddings), employee_id,
    )
    return {
        "success": True,
        "message": f"{len(new_embeddings)} new image(s) added for '{name}'.",
        "employee_id": employee_id,
        "embeddings_saved": len(new_embeddings),
        "images_saved": len(new_embeddings),
        "skipped": skipped,
    }


def _crop_largest_face(
    img: np.ndarray,
    min_confidence: float = 0.5,
) -> np.ndarray | None:
    """
    Detect faces in img using OpenCV DNN, return the largest face crop.
    Falls back to Haar cascade if DNN model files are unavailable.
    If no face is found (image is already a tight crop), return img as-is
    only when it looks like a face-sized region (height > 50 px).
    """
    h_img, w_img = img.shape[:2]

    # --- DNN detector ---
    try:
        weights_dir = get_config().embeddings_dir.parent / "weights"
        prototxt   = weights_dir / "deploy.prototxt"
        caffemodel = weights_dir / "res10_300x300_ssd_iter_140000.caffemodel"

        if prototxt.exists() and caffemodel.exists():
            net  = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
            blob = cv2.dnn.blobFromImage(
                cv2.resize(img, (300, 300)), 1.0, (300, 300), (104, 177, 123)
            )
            net.setInput(blob)
            detections = net.forward()

            best_box, best_conf = None, 0.0
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf < min_confidence:
                    continue
                box = detections[0, 0, i, 3:7] * np.array([w_img, h_img, w_img, h_img])
                x1, y1, x2, y2 = box.astype(int)
                area = (x2 - x1) * (y2 - y1)
                if area > best_conf:
                    best_conf = area
                    best_box  = (x1, y1, x2, y2)

            if best_box:
                x1, y1, x2, y2 = best_box
                pad = 20
                x1 = max(0, x1 - pad);  y1 = max(0, y1 - pad)
                x2 = min(w_img, x2 + pad); y2 = min(h_img, y2 + pad)
                return img[y1:y2, x1:x2]
    except Exception as e:
        log.debug("DNN detection failed ({}), trying Haar cascade", e)

    # --- Haar cascade fallback ---
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces   = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces):
            # Largest face by area
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            pad = 20
            x1 = max(0, x - pad);  y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad); y2 = min(h_img, y + h + pad)
            return img[y1:y2, x1:x2]
    except Exception as e:
        log.debug("Haar detection failed: {}", e)

    # --- Assume the image is already a face crop ---
    if h_img >= 50 and w_img >= 50:
        log.debug("No face detected — treating image as pre-cropped face")
        return img

    return None


def _extract_embedding(face_crop: np.ndarray, model_name: str) -> np.ndarray | None:
    """
    Run DeepFace.represent() on a cropped face image.
    Returns a float32 numpy array (512-D for Facenet512), or None on failure.
    """
    try:
        result = DeepFace.represent(
            img_path=face_crop,
            model_name=model_name,
            detector_backend="skip",   # already cropped — skip internal detection
            enforce_detection=False,
        )
        return np.array(result[0]["embedding"], dtype=np.float32)
    except Exception as e:
        log.warning("DeepFace.represent failed: {}", e)
        return None


def _generate_id(name: str) -> str:
    """
    Auto-generate a unique employee ID from name + 4-char UUID suffix.
    e.g. "Jane Smith" → "EMP_JaneSmith_a3f1"
    """
    slug = re.sub(r"\s+", "", name.title())[:20]        # "JaneSmith"
    suffix = uuid.uuid4().hex[:4]                        # "a3f1"
    return f"EMP_{slug}_{suffix}"


def _slug(name: str) -> str:
    """Convert "Jane Smith" → "jane.smith" for placeholder email."""
    return re.sub(r"\s+", ".", name.strip().lower())


def _fail(message: str, employee_id: str = "") -> dict:
    log.error("Registration failed: {}", message)
    return {
        "success": False,
        "message": message,
        "employee_id": employee_id,
        "embeddings_saved": 0,
        "images_saved": 0,
        "skipped": 0,
    }

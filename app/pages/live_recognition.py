"""
Page: Live Recognition

Architecture
────────────
  CaptureThread    : reads camera at full speed → _raw_frame (shared)
  RecognitionThread: reads _raw_frame → runs DeepFace → stores _results (shared)
  Display loop     : reads _raw_frame + _results → draws boxes → JPEG → st.empty()

Why this is smooth
──────────────────
  • Display always uses the LATEST raw frame — never waits for recognition.
  • Bounding boxes from the last recognition pass are overlaid every display tick.
  • JPEG encoding (quality 80) cuts payload size ~10× vs raw numpy → fast browser render.
  • Recognition runs on a downscaled frame (320×240) → ~5 fps on CPU, non-blocking.
  • Only ONE st.empty() widget is updated per tick — zero double-repaint flicker.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import threading
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Live Recognition", layout="wide")

from app.components.sidebar import render_sidebar
from backend.embedding_manager import EmbeddingManager
from backend.face_detection import FaceDetector
from backend.face_recognition import FaceRecognizer, RecognitionResult
from database.db_handler import get_db

render_sidebar()

st.markdown('<div class="section-header">Live Face Recognition</div>',
            unsafe_allow_html=True)

# ── Cached heavy resources ────────────────────────────────────────────────────
@st.cache_resource
def _get_recognizer():
    emb_mgr = EmbeddingManager()
    detector = FaceDetector()
    recognizer = FaceRecognizer(emb_mgr)
    return detector, recognizer


# ── Session state ─────────────────────────────────────────────────────────────
for key, val in [("cam_running", False), ("session_marked", set()), ("att_log", [])]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Layout ────────────────────────────────────────────────────────────────────
col_cam, col_log = st.columns([3, 1])

with col_log:
    st.markdown('<div class="section-header">Controls</div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    start_btn = b1.button("Start", type="primary", use_container_width=True)
    stop_btn  = b2.button("Stop",                  use_container_width=True)
    st.markdown("---")
    st.markdown('<div class="section-header">Session Log</div>', unsafe_allow_html=True)
    log_box = st.empty()

with col_cam:
    frame_box = st.empty()

# ── Idle placeholder ──────────────────────────────────────────────────────────
if not st.session_state.cam_running:
    frame_box.markdown(
        """<div style="background:#1e2130;border:2px dashed #2a2f45;border-radius:12px;
           height:420px;display:flex;align-items:center;justify-content:center;
           color:#4a5068;font-size:1.1rem;">Camera is off — press Start</div>""",
        unsafe_allow_html=True,
    )

# ── Button handlers ───────────────────────────────────────────────────────────
if start_btn and not st.session_state.cam_running:
    st.session_state.cam_running = True
    st.rerun()

if stop_btn and st.session_state.cam_running:
    st.session_state.cam_running = False
    st.rerun()

# ── Main camera system ────────────────────────────────────────────────────────
if not st.session_state.cam_running:
    st.stop()

db        = get_db()
detector, recognizer = _get_recognizer()

# Shared buffers (lock-protected)
_lock      = threading.Lock()
_stop      = threading.Event()
_raw       = [None]          # latest BGR frame from camera
_results   = [[]]            # latest list[RecognitionResult]

# ── Thread 1: capture ─────────────────────────────────────────────────────────
def _capture_loop(cap):
    while not _stop.is_set():
        ret, frame = cap.read()
        if ret:
            with _lock:
                _raw[0] = frame
        else:
            time.sleep(0.005)

# ── Thread 2: recognition (runs on downscaled frame for speed) ────────────────
def _recognition_loop():
    REC_W, REC_H = 320, 240
    while not _stop.is_set():
        with _lock:
            frame = _raw[0]
        if frame is None:
            time.sleep(0.02)
            continue

        h, w = frame.shape[:2]
        small = cv2.resize(frame, (REC_W, REC_H))
        sx, sy = w / REC_W, h / REC_H

        try:
            faces = detector.detect(small)
            results = []
            for face in faces:
                crop = face.as_crop(small)
                if crop.size == 0:
                    continue
                r = recognizer.recognize(crop, bbox=(face.x, face.y, face.w, face.h))
                # Scale bbox back to display resolution
                if r.bbox:
                    bx, by, bw, bh = r.bbox
                    scaled_bbox = (int(bx*sx), int(by*sy), int(bw*sx), int(bh*sy))
                    r = RecognitionResult(
                        employee_id=r.employee_id, name=r.name,
                        confidence=r.confidence, matched=r.matched,
                        bbox=scaled_bbox,
                    )
                results.append(r)
            with _lock:
                _results[0] = results
        except Exception:
            pass

        time.sleep(0.05)   # recognition rate ~20fps max, usually 3-8fps on CPU

# ── Draw boxes on frame ───────────────────────────────────────────────────────
def _draw(frame: np.ndarray, results: list[RecognitionResult]) -> bytes:
    out = frame.copy()
    for r in results:
        if not r.bbox:
            continue
        x, y, w, h = r.bbox
        color = (0, 200, 0) if r.matched else (0, 0, 220)
        label = f"{r.name} | Employee" if r.matched else "Not Employee"
        conf  = f"{r.confidence:.0%}"

        # Box
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)
        # Corner accents
        cl = 16
        for (px, py, dx, dy) in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
            cv2.line(out, (px, py), (px+dx*cl, py), color, 3)
            cv2.line(out, (px, py), (px, py+dy*cl), color, 3)
        # Label background
        (tw, th), _ = cv2.getTextSize(f"{label} {conf}", cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
        cv2.rectangle(out, (x, y-th-10), (x+tw+6, y), (0,0,0), -1)
        cv2.putText(out, f"{label} {conf}", (x+3, y-5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Encode to JPEG — ~10× smaller than raw numpy, renders instantly in browser
    _, jpeg = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg.tobytes()

# ── Open camera ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Cannot open camera — make sure no other app is using it.")
    st.session_state.cam_running = False
    st.stop()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          30)
cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

# ── Start threads ─────────────────────────────────────────────────────────────
t1 = threading.Thread(target=_capture_loop,    args=(cap,), daemon=True)
t2 = threading.Thread(target=_recognition_loop,              daemon=True)
t1.start()
t2.start()

# ── Display loop ──────────────────────────────────────────────────────────────
prev_log_len = 0
try:
    while st.session_state.cam_running:
        with _lock:
            frame   = _raw[0]
            results = list(_results[0])

        if frame is not None:
            # Draw boxes + encode — single image update per tick
            jpeg = _draw(frame, results)
            frame_box.image(jpeg, use_column_width=True)

            # Mark attendance for newly recognised employees
            for r in results:
                if (r.matched
                        and r.employee_id
                        and r.employee_id not in st.session_state.session_marked):
                    try:
                        db.mark_attendance(r.employee_id, r.name, r.confidence)
                        st.session_state.session_marked.add(r.employee_id)
                        st.session_state.att_log.append({
                            "name":       r.name,
                            "time":       datetime.now().strftime("%H:%M:%S"),
                            "confidence": f"{r.confidence:.0%}",
                        })
                    except Exception as e:
                        st.warning(f"Attendance error for {r.name}: {e}")

            # Update log only when new entries added
            if len(st.session_state.att_log) != prev_log_len:
                prev_log_len = len(st.session_state.att_log)
                log_box.dataframe(
                    pd.DataFrame(st.session_state.att_log),
                    use_container_width=True, hide_index=True,
                )

        time.sleep(0.04)   # ~25 fps display

finally:
    _stop.set()
    t1.join(timeout=1)
    t2.join(timeout=2)
    cap.release()
    st.session_state.cam_running = False

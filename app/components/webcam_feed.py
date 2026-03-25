"""
webcam_feed.py — Webcam snapshot component for registration page.
(Live recognition now uses direct OpenCV loop in live_recognition.py)
"""
from __future__ import annotations

import cv2
import numpy as np
import streamlit as st


def render_snapshot() -> np.ndarray | None:
    """
    Camera snapshot widget — returns BGR numpy array when photo taken, else None.
    Used on the Register Employee page.
    """
    img_file = st.camera_input("Take a photo")
    if img_file is not None:
        arr = np.frombuffer(img_file.read(), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return None

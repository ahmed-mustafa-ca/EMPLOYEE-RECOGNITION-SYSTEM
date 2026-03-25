"""
Page: Register Employee
Enroll a new employee with face photos (upload or webcam snapshot).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import numpy as np
import streamlit as st

st.set_page_config(page_title="Register Employee", layout="wide")

from app.components.sidebar import render_sidebar
from app.components.webcam_feed import render_snapshot
from backend.registration import register_employee
from database.db_handler import get_db

render_sidebar()

st.markdown('<div class="section-header">Register New Employee</div>', unsafe_allow_html=True)

# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("register_form", clear_on_submit=True):
    c1, c2 = st.columns(2)
    with c1:
        name    = st.text_input("Full Name *", placeholder="e.g. Ahmed Mustafa")
        dept    = st.text_input("Department",  placeholder="e.g. Engineering")
    with c2:
        emp_id  = st.text_input("Employee ID (leave blank to auto-generate)", placeholder="e.g. EMP001")
        email   = st.text_input("Email",       placeholder="e.g. ahmed@company.com")

    st.markdown('<div class="section-header">Face Photos</div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">Upload 3–5 clear, well-lit, front-facing photos for best accuracy.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload images", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, label_visibility="collapsed",
    )

    submitted = st.form_submit_button("Register Employee", type="primary", use_container_width=True)

# ── Preview uploaded images ───────────────────────────────────────────────────
if uploaded:
    st.markdown('<div class="section-header">Preview</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(uploaded), 5))
    for col, f in zip(cols, uploaded[:5]):
        col.image(f, width=200, caption=f.name)

# ── Webcam snapshot tab ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">Or Use Webcam Snapshot</div>', unsafe_allow_html=True)
snapshot_img = render_snapshot()

# ── On submit ─────────────────────────────────────────────────────────────────
if submitted:
    if not name.strip():
        st.error("Full Name is required.")
    else:
        images: list[np.ndarray] = []

        # From file uploader
        for f in (uploaded or []):
            arr = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                images.append(img)

        # From webcam snapshot
        if snapshot_img is not None:
            images.append(snapshot_img)

        if not images:
            st.error("Please upload at least one photo or take a webcam snapshot.")
        else:
            with st.spinner("Extracting face embeddings and saving…"):
                try:
                    result = register_employee(
                        name=name.strip(),
                        images=images,
                        employee_id=emp_id.strip() or None,
                        department=dept.strip() or "",
                        email=email.strip() or "",
                    )
                except Exception as e:
                    st.error(f"Unexpected error during registration: {e}")
                    st.stop()

            if result["success"]:
                st.success(
                    f"Employee **{name}** registered  "
                    f"(ID: `{result['employee_id']}`).  "
                    f"{result['embeddings_saved']} embedding(s) saved."
                )
                if result.get("skipped", 0):
                    st.warning(f"{result['skipped']} image(s) skipped — no face detected.")
            else:
                st.error(result.get("message", "Registration failed."))

# ── Registered employees table ────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Registered Employees</div>', unsafe_allow_html=True)

try:
    db = get_db()
    employees = db.get_all_employees()
    if employees:
        import pandas as pd
        df = pd.DataFrame(employees)
        cols_show = [c for c in ["employee_id", "name", "department", "email", "created_at"] if c in df.columns]
        st.dataframe(df[cols_show], use_container_width=True, hide_index=True)
    else:
        st.info("No employees registered yet.")
except Exception as e:
    st.error(f"Could not load employee list: {e}")

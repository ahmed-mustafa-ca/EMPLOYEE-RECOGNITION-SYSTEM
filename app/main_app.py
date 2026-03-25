"""
Entry point — streamlit run app/main_app.py
"""
import sys
from pathlib import Path
# Ensure project root is on sys.path so all packages resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Employee Recognition System",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)

from datetime import date, timedelta
from app.components.sidebar import render_sidebar, inject_css
from database.db_handler import get_db

inject_css()
render_sidebar()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-card">
        <h1>Employee Recognition System</h1>
        <p>AI-powered attendance tracking using real-time face recognition.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Live KPI cards ────────────────────────────────────────────────────────────
db = get_db()
try:
    all_emp      = db.get_all_employees()
    today_att    = db.get_attendance_by_date(date.today())
    week_att     = db.get_attendance_range(date.today() - timedelta(days=6), date.today())
    total_emp     = len(all_emp)
    present_today = len(today_att)
    absent_today  = max(0, total_emp - present_today)
except Exception:
    total_emp = present_today = absent_today = 0

c1, c2, c3 = st.columns(3)
c1.metric("Total Employees", total_emp)
c2.metric("Present Today",   present_today)
c3.metric("Absent Today",    absent_today)

st.markdown("---")

# ── Feature cards ────────────────────────────────────────────────────────────
st.subheader("Modules")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """<div class="feature-card">
            <div class="feature-icon">🎥</div>
            <h3>Live Recognition</h3>
            <p>Real-time webcam feed with automatic face detection and attendance marking.</p>
        </div>""",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """<div class="feature-card">
            <div class="feature-icon">➕</div>
            <h3>Register Employee</h3>
            <p>Enroll new employees by uploading photos or using the webcam snapshot.</p>
        </div>""",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """<div class="feature-card">
            <div class="feature-icon">📋</div>
            <h3>Attendance</h3>
            <p>View, filter, and export full attendance records by date or employee.</p>
        </div>""",
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """<div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3>Dashboard</h3>
            <p>KPIs, weekly trends, and department breakdowns at a glance.</p>
        </div>""",
        unsafe_allow_html=True,
    )

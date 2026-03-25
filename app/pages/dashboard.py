"""
Page: Dashboard
KPI cards, weekly attendance trend, department breakdown, employee table.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Dashboard", layout="wide")

from datetime import date, timedelta

import pandas as pd

from app.components.sidebar import render_sidebar
from database.db_handler import get_db

render_sidebar()

st.markdown('<div class="section-header">Dashboard</div>', unsafe_allow_html=True)

db    = get_db()
today = date.today()

# ── Fetch data ────────────────────────────────────────────────────────────────
try:
    all_emp    = db.get_all_employees()
    today_att  = db.get_attendance_by_date(today)
    week_att   = db.get_attendance_range(today - timedelta(days=6), today)
except Exception as e:
    st.error(f"Database error: {e}")
    st.stop()

# Build employee lookup for department enrichment
emp_map = {e["employee_id"]: e.get("department", "—") for e in all_emp}

# Enrich attendance with department
for r in today_att:
    r["department"] = emp_map.get(r["employee_id"], "—")
for r in week_att:
    r["department"] = emp_map.get(r["employee_id"], "—")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
total_emp     = len(all_emp)
present_today = len(today_att)
absent_today  = max(0, total_emp - present_today)

k1, k2, k3 = st.columns(3)
k1.metric("Total Employees", total_emp)
k2.metric("Present Today",   present_today,
          delta=f"{present_today/total_emp:.0%} attendance" if total_emp else None)
k3.metric("Absent Today",    absent_today)

st.markdown("---")

# ── Charts row ────────────────────────────────────────────────────────────────
chart_l, chart_r = st.columns(2)

# Weekly trend
with chart_l:
    st.markdown('<div class="section-header">Weekly Attendance Trend</div>', unsafe_allow_html=True)
    if week_att:
        df_week = pd.DataFrame(week_att)
        df_week["date"] = pd.to_datetime(df_week["date"])
        daily = df_week.groupby("date")["employee_id"].nunique().reset_index()
        daily.columns = ["date", "Present"]
        st.bar_chart(daily.set_index("date"), color="#4f8ef7")
    else:
        st.info("No attendance data this week.")

# Department breakdown for today
with chart_r:
    st.markdown('<div class="section-header">Department Presence Today</div>', unsafe_allow_html=True)
    if today_att:
        df_today = pd.DataFrame(today_att)
        dept_counts = df_today["department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Present"]
        st.bar_chart(dept_counts.set_index("Department"), color="#00d48b")
    else:
        st.info("No attendance data today.")

st.markdown("---")

st.markdown("---")

# ── Employee table ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">All Employees</div>', unsafe_allow_html=True)
if all_emp:
    df_emp = pd.DataFrame(all_emp)
    show   = [c for c in ["employee_id", "name", "department", "email", "is_active", "created_at"] if c in df_emp.columns]
    st.dataframe(
        df_emp[show],
        use_container_width=True,
        hide_index=True,
        column_config={
            "is_active": st.column_config.CheckboxColumn("Active"),
            "created_at": st.column_config.TextColumn("Registered"),
        },
    )
else:
    st.info("No employees registered yet.")

# ── Delete Employee ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Delete Employee</div>', unsafe_allow_html=True)
st.markdown('<div class="info-box">Deleting an employee also removes all their attendance records.</div>',
            unsafe_allow_html=True)

if all_emp:
    emp_options = {f"{e['name']} ({e['employee_id']})": e["employee_id"] for e in all_emp}
    selected = st.selectbox("Select employee to delete", list(emp_options.keys()))

    if st.button("Delete Employee", type="primary"):
        st.session_state["confirm_emp_delete"] = emp_options[selected]

    if st.session_state.get("confirm_emp_delete"):
        eid = st.session_state["confirm_emp_delete"]
        st.warning(f"Permanently delete **{selected}** and all their attendance records?")
        c1, c2 = st.columns(2)
        if c1.button("Yes, delete", key="emp_del_yes"):
            deleted = db.delete_employee(eid)
            if deleted:
                st.success(f"Employee **{selected}** deleted.")
            else:
                st.error("Could not delete employee.")
            st.session_state.pop("confirm_emp_delete", None)
            st.rerun()
        if c2.button("Cancel", key="emp_del_no"):
            st.session_state.pop("confirm_emp_delete", None)
            st.rerun()
else:
    st.info("No employees to delete.")

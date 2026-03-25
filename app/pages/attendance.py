"""
Page: Attendance Records
View, filter, and export attendance data.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Attendance", layout="wide")

from datetime import date, timedelta

import pandas as pd

from app.components.sidebar import render_sidebar
from database.db_handler import get_db

render_sidebar()

st.markdown('<div class="section-header">Attendance Records</div>', unsafe_allow_html=True)

db = get_db()

# ── Filters ───────────────────────────────────────────────────────────────────
f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
with f1:
    from_date = st.date_input("From", value=date.today() - timedelta(days=7))
with f2:
    to_date = st.date_input("To", value=date.today())
with f3:
    name_filter = st.text_input("Search name", placeholder="e.g. Ahmed")
with f4:
    status_filter = st.selectbox("Status", ["All", "present"])

load = st.button("Load Records", type="primary", use_container_width=False)

st.markdown("---")

if load or "att_records" not in st.session_state:
    try:
        records = db.get_attendance_range(from_date, to_date)

        # Enrich with department
        emp_map = {e["employee_id"]: e.get("department", "—") for e in db.get_all_employees()}
        for r in records:
            r["department"] = emp_map.get(r["employee_id"], "—")

        st.session_state["att_records"] = records
        st.session_state["att_from"]    = from_date
        st.session_state["att_to"]      = to_date
    except Exception as e:
        st.error(f"Database error: {e}")
        st.stop()

records = st.session_state.get("att_records", [])

# ── Client-side filters ───────────────────────────────────────────────────────
if name_filter:
    records = [r for r in records if name_filter.lower() in r.get("name", "").lower()]
if status_filter != "All":
    records = [r for r in records if r.get("status") == status_filter]

# ── Results table ─────────────────────────────────────────────────────────────
if records:
    df = pd.DataFrame(records)

    # Rename columns for display
    df = df.rename(columns={
        "time":             "check_in",
        "confidence_score": "confidence",
    })

    # Format
    df["check_in"]  = df["check_in"].astype(str)
    df["check_out"] = df["check_out"].astype(str).replace("None", "—") if "check_out" in df.columns else "—"
    df["confidence"] = df["confidence"].apply(lambda x: f"{float(x):.1%}" if x else "—")

    show_cols = [c for c in
        ["id", "name", "department", "date", "check_in", "check_out", "confidence", "status"]
        if c in df.columns]

    st.markdown(f"**{len(df)} record(s)** found between `{from_date}` and `{to_date}`")

    st.dataframe(
        df[show_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "id":          st.column_config.NumberColumn("ID",        width="small"),
            "name":        st.column_config.TextColumn("Name"),
            "department":  st.column_config.TextColumn("Dept"),
            "date":        st.column_config.TextColumn("Date",        width="small"),
            "check_in":    st.column_config.TextColumn("Check-In",    width="small"),
            "check_out":   st.column_config.TextColumn("Check-Out",   width="small"),
            "confidence":  st.column_config.TextColumn("Confidence",  width="small"),
            "status":      st.column_config.TextColumn("Status",      width="small"),
        },
    )

    # ── Export ────────────────────────────────────────────────────────────────
    csv = df[show_cols].to_csv(index=False)
    st.download_button(
        label="Export CSV",
        data=csv,
        file_name=f"attendance_{from_date}_{to_date}.csv",
        mime="text/csv",
    )
else:
    st.info("No records found. Adjust filters and click Load Records.")

# ── Delete Section ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">Delete Attendance Records</div>', unsafe_allow_html=True)

del_tab1, del_tab2, del_tab3 = st.tabs(["By Record ID", "By Employee + Date", "By Date Range"])

# ── Tab 1: Delete single record by ID ────────────────────────────────────────
with del_tab1:
    st.markdown('<div class="info-box">Delete one specific record using its ID from the table above.</div>',
                unsafe_allow_html=True)
    record_id = st.number_input("Record ID", min_value=1, step=1, key="del_id")
    if st.button("Delete Record", type="primary", key="del_by_id"):
        deleted = db.delete_attendance_by_id(int(record_id))
        if deleted:
            st.success(f"Record ID {int(record_id)} deleted.")
            st.session_state.pop("att_records", None)
            st.rerun()
        else:
            st.error(f"No record found with ID {int(record_id)}.")

# ── Tab 2: Delete by employee + date ─────────────────────────────────────────
with del_tab2:
    st.markdown('<div class="info-box">Delete the attendance entry for a specific employee on a specific date.</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        del_emp_id = st.text_input("Employee ID", placeholder="e.g. EMP001", key="del_emp")
    with c2:
        del_date = st.date_input("Date", value=date.today(), key="del_date")

    if st.button("Delete Record", type="primary", key="del_by_emp"):
        if not del_emp_id.strip():
            st.error("Employee ID is required.")
        else:
            deleted = db.delete_attendance_by_employee_date(del_emp_id.strip(), del_date)
            if deleted:
                st.success(f"Attendance for **{del_emp_id}** on **{del_date}** deleted.")
                st.session_state.pop("att_records", None)
                st.rerun()
            else:
                st.error("No matching record found.")

# ── Tab 3: Delete by date range ───────────────────────────────────────────────
with del_tab3:
    st.markdown('<div class="info-box">Delete all attendance records within a date range. This cannot be undone.</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        del_from = st.date_input("From", value=date.today(), key="del_from")
    with c2:
        del_to = st.date_input("To", value=date.today(), key="del_to")

    if st.button("Delete All in Range", type="primary", key="del_range"):
        if del_from > del_to:
            st.error("'From' date must be before 'To' date.")
        else:
            # Confirmation via session state
            st.session_state["confirm_range_delete"] = True

    if st.session_state.get("confirm_range_delete"):
        st.warning(f"Delete ALL records from **{del_from}** to **{del_to}**?")
        cc1, cc2 = st.columns(2)
        if cc1.button("Yes, delete", key="confirm_yes"):
            count = db.delete_attendance_range(del_from, del_to)
            st.success(f"{count} record(s) deleted.")
            st.session_state.pop("confirm_range_delete", None)
            st.session_state.pop("att_records", None)
            st.rerun()
        if cc2.button("Cancel", key="confirm_no"):
            st.session_state.pop("confirm_range_delete", None)
            st.rerun()

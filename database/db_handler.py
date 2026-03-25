"""
db_handler.py — SQLite data-access layer for the Employee Recognition System.

Design decisions
────────────────
• Every public method opens its own session via a context manager and closes
  it on exit, keeping connections short-lived and thread-safe.
• SQLAlchemy's check_same_thread=False lets Streamlit's threaded re-runs
  share the same engine without errors.
• The singleton get_db() is safe to call from anywhere; it returns the same
  DBHandler instance for the lifetime of the process.
• Duplicate attendance is blocked at two levels:
    1. DB  : UNIQUE(employee_id, date) — raises IntegrityError on conflict.
    2. App : AttendanceManager cooldown dict (see backend/attendance_manager.py).
"""
from __future__ import annotations

import pickle
from datetime import date, datetime, time, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker, Session

from database.models import Base, Employee, AttendanceRecord
from utils.config import get_config
from utils.logger import get_logger

log = get_logger(__name__)


class DBHandler:
    """Thread-safe SQLite data-access layer."""

    def __init__(self, db_path: Path | None = None):
        cfg = get_config()
        path = db_path or cfg.db_path
        path.parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_engine(
            f"sqlite:///{path}",
            connect_args={"check_same_thread": False},
            echo=False,
        )
        # Enable WAL mode for concurrent read/write performance
        with self._engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            conn.exec_driver_sql("PRAGMA foreign_keys=ON")

        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False)
        log.info("Database ready at '{}'", path)

    def _session(self) -> Session:
        return self._Session()

    # =========================================================================
    # Employee CRUD
    # =========================================================================

    def add_employee(
        self,
        employee_id: str,
        name: str,
        department: str,
        email: str,
        image_path: str | None = None,
        embedding: np.ndarray | None = None,
    ) -> Employee:
        """
        Insert a new employee row.

        Parameters
        ----------
        embedding : numpy float32 array (e.g. shape (512,) for Facenet512).
                    Serialised to bytes with pickle before storage.
                    Retrieve with get_employee_embedding().
        """
        blob = pickle.dumps(embedding) if embedding is not None else None
        with self._session() as s:
            emp = Employee(
                employee_id=employee_id,
                name=name,
                department=department,
                email=email,
                image_path=image_path,
                embedding=blob,
            )
            s.add(emp)
            s.commit()
            log.debug("Added employee '{}'", employee_id)
            return emp

    def get_employee(self, employee_id: str) -> dict | None:
        with self._session() as s:
            emp = s.get(Employee, employee_id)
            return emp.to_dict() if emp else None

    def get_employee_embedding(self, employee_id: str) -> np.ndarray | None:
        """Return the deserialised embedding vector, or None if not stored."""
        with self._session() as s:
            emp = s.get(Employee, employee_id)
            if emp is None or emp.embedding is None:
                return None
            return pickle.loads(emp.embedding)

    def get_all_employees(self, active_only: bool = True) -> list[dict]:
        with self._session() as s:
            q = s.query(Employee)
            if active_only:
                q = q.filter(Employee.is_active == True)   # noqa: E712
            return [e.to_dict() for e in q.order_by(Employee.name).all()]

    def employee_exists(self, employee_id: str) -> bool:
        with self._session() as s:
            return s.get(Employee, employee_id) is not None

    def update_employee(self, employee_id: str, **fields) -> bool:
        """
        Update allowed fields on an employee row.
        Pass embedding=<np.ndarray> to replace the stored vector.
        """
        allowed = {"name", "department", "email", "image_path", "is_active"}
        with self._session() as s:
            emp = s.get(Employee, employee_id)
            if not emp:
                return False
            for key, val in fields.items():
                if key == "embedding" and isinstance(val, np.ndarray):
                    emp.embedding = pickle.dumps(val)
                elif key in allowed:
                    setattr(emp, key, val)
            emp.updated_at = datetime.now(timezone.utc)
            s.commit()
            log.debug("Updated employee '{}'", employee_id)
            return True

    def delete_employee(self, employee_id: str) -> bool:
        """Hard-delete an employee and all their attendance records (CASCADE)."""
        with self._session() as s:
            emp = s.get(Employee, employee_id)
            if not emp:
                return False
            s.delete(emp)
            s.commit()
            log.info("Deleted employee '{}'", employee_id)
            return True

    # =========================================================================
    # Attendance
    # =========================================================================

    def mark_attendance(
        self,
        employee_id: str,
        name: str,
        confidence_score: float,
    ) -> dict:
        """
        Mark attendance for today.

        Duplicate handling
        ──────────────────
        • First recognition of the day → INSERT a new row (check-in).
        • Subsequent recognitions the same day → UPDATE check_out only.
          The UNIQUE(employee_id, date) constraint guarantees one row per day.

        Returns the attendance record as a dict.
        """
        today     = date.today()
        now_time  = datetime.now().time().replace(microsecond=0)

        with self._session() as s:
            existing = (
                s.query(AttendanceRecord)
                .filter(
                    AttendanceRecord.employee_id == employee_id,
                    AttendanceRecord.date == today,
                )
                .first()
            )

            if existing:
                # Already checked in today — just update check_out
                existing.check_out = now_time
                s.commit()
                log.debug("Updated check_out for '{}' @ {}", employee_id, now_time)
                return existing.to_dict()

            # First recognition today — create check-in record
            record = AttendanceRecord(
                employee_id=employee_id,
                name=name,
                date=today,
                time=now_time,
                confidence_score=round(confidence_score, 4),
                status=self._compute_status(now_time),
            )
            try:
                s.add(record)
                s.commit()
                log.info("Check-in: '{}' ({}) @ {} [conf={:.2%}]",
                         name, employee_id, now_time, confidence_score)
                return record.to_dict()
            except IntegrityError:
                # Race condition: another thread inserted between our SELECT and INSERT
                s.rollback()
                existing = (
                    s.query(AttendanceRecord)
                    .filter(
                        AttendanceRecord.employee_id == employee_id,
                        AttendanceRecord.date == today,
                    )
                    .first()
                )
                return existing.to_dict() if existing else {}

    def get_attendance_by_date(self, target_date: date) -> list[dict]:
        """All attendance records for a given calendar date."""
        with self._session() as s:
            rows = (
                s.query(AttendanceRecord)
                .filter(AttendanceRecord.date == target_date)
                .order_by(AttendanceRecord.time)
                .all()
            )
            return [r.to_dict() for r in rows]

    def get_attendance_by_employee(self, employee_id: str) -> list[dict]:
        """Full attendance history for one employee."""
        with self._session() as s:
            rows = (
                s.query(AttendanceRecord)
                .filter(AttendanceRecord.employee_id == employee_id)
                .order_by(AttendanceRecord.date.desc())
                .all()
            )
            return [r.to_dict() for r in rows]

    def get_attendance_range(self, from_date: date, to_date: date) -> list[dict]:
        """Attendance records between two dates (inclusive)."""
        with self._session() as s:
            rows = (
                s.query(AttendanceRecord)
                .filter(
                    AttendanceRecord.date >= from_date,
                    AttendanceRecord.date <= to_date,
                )
                .order_by(AttendanceRecord.date, AttendanceRecord.time)
                .all()
            )
            return [r.to_dict() for r in rows]

    def delete_attendance_by_id(self, record_id: int) -> bool:
        """Delete a single attendance record by its primary key. Returns True if deleted."""
        try:
            with self._session() as s:
                row = s.query(AttendanceRecord).filter(AttendanceRecord.id == record_id).first()
                if row:
                    s.delete(row)
                    s.commit()
                    log.info("Deleted attendance record id={}", record_id)
                    return True
                return False
        except Exception as e:
            log.error("delete_attendance_by_id failed (id={}): {}", record_id, e)
            return False

    def delete_attendance_by_employee_date(self, employee_id: str, target_date: date) -> bool:
        """Delete the attendance record for one employee on one date."""
        try:
            with self._session() as s:
                row = s.query(AttendanceRecord).filter(
                    AttendanceRecord.employee_id == employee_id,
                    AttendanceRecord.date == target_date,
                ).first()
                if row:
                    s.delete(row)
                    s.commit()
                    log.info("Deleted attendance for '{}' on {}", employee_id, target_date)
                    return True
                return False
        except Exception as e:
            log.error("delete_attendance_by_employee_date failed: {}", e)
            return False

    def delete_attendance_range(self, from_date: date, to_date: date) -> int:
        """Delete all attendance records between two dates. Returns count deleted."""
        try:
            with self._session() as s:
                rows = s.query(AttendanceRecord).filter(
                    AttendanceRecord.date >= from_date,
                    AttendanceRecord.date <= to_date,
                ).all()
                count = len(rows)
                for row in rows:
                    s.delete(row)
                s.commit()
                log.info("Deleted {} attendance records ({} → {})", count, from_date, to_date)
                return count
        except Exception as e:
            log.error("delete_attendance_range failed: {}", e)
            return 0

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _compute_status(check_in: time) -> str:
        return "present"


@lru_cache(maxsize=1)
def get_db() -> DBHandler:
    """Module-level singleton — safe to call from anywhere."""
    return DBHandler()

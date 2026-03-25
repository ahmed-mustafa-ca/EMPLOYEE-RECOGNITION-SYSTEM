"""
SQLAlchemy ORM models — single source of truth for the database schema.
Schema is auto-created on first run via Base.metadata.create_all() in db_handler.py.
"""
from datetime import datetime, date, time, timezone

from sqlalchemy import (
    Column, String, Float, DateTime, Date, Time,
    Boolean, Integer, Text, LargeBinary, ForeignKey, UniqueConstraint, Index,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Employee(Base):
    """
    One row per enrolled employee.

    Columns
    -------
    employee_id  : business-level ID  (e.g. "EMP001") — primary key
    name         : full display name
    image_path   : relative path to the representative face image on disk
                   e.g. "data/employee_images/EMP001/000.jpg"
    embedding    : face embedding vector serialised as raw bytes (pickle/numpy)
                   stored as BLOB; deserialise with pickle.loads() or np.frombuffer()
    department   : organisational unit
    email        : unique contact email
    is_active    : soft-delete flag — False hides the employee from recognition
    created_at   : row creation timestamp (UTC)
    updated_at   : last modification timestamp (UTC)
    """
    __tablename__ = "employees"

    employee_id = Column(String(20),   primary_key=True)
    name        = Column(String(60),   nullable=False)
    image_path  = Column(String(255),  nullable=True)   # path to primary face image
    embedding   = Column(LargeBinary,  nullable=True)   # serialised numpy float32 array
    department  = Column(String(50),   nullable=False, default="")
    email       = Column(String(120),  nullable=False, unique=True)
    is_active   = Column(Boolean,      nullable=False, default=True)
    created_at  = Column(DateTime,     nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at  = Column(DateTime,     nullable=False, default=lambda: datetime.now(timezone.utc),
                         onupdate=lambda: datetime.now(timezone.utc))

    attendance_records = relationship(
        "AttendanceRecord",
        back_populates="employee",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> dict:
        return {
            "employee_id": self.employee_id,
            "name":        self.name,
            "image_path":  self.image_path,
            "department":  self.department,
            "email":       self.email,
            "is_active":   self.is_active,
            "created_at":  self.created_at.isoformat() if self.created_at else None,
        }
        # NOTE: embedding is intentionally excluded from to_dict —
        #       it is a large binary blob and should be fetched explicitly.


class AttendanceRecord(Base):
    """
    One row per employee per calendar day.

    Duplicate prevention strategy (two layers):
    ─────────────────────────────────────────────
    1. DB constraint  : UNIQUE(employee_id, date) — enforced by SQLite itself.
       Any INSERT that would create a second row for the same employee on the
       same day raises IntegrityError. The handler catches this and updates
       check_out instead (upsert pattern).

    2. Application cooldown : AttendanceManager keeps an in-memory dict
       {employee_id: last_marked_at} and skips marking within the cooldown
       window (default 60 s). This prevents hammering the DB on every frame.

    Columns
    -------
    id               : auto-increment surrogate key
    employee_id      : FK → employees.employee_id  (CASCADE delete)
    name             : denormalised copy of employee name — avoids a JOIN on
                       every attendance query; updated if employee name changes
    date             : calendar date of attendance  (DATE)
    time             : wall-clock time of first recognition today  (TIME)
    check_out        : wall-clock time of last recognition today   (TIME, nullable)
    confidence_score : cosine similarity at check-in (0.0 – 1.0)
    status           : 'present' | 'late' | 'early_leave'
    notes            : optional free-text annotation
    """
    __tablename__ = "attendance"

    id               = Column(Integer,    primary_key=True, autoincrement=True)
    employee_id      = Column(String(20), ForeignKey("employees.employee_id",
                              ondelete="CASCADE"), nullable=False)
    name             = Column(String(60), nullable=False)         # denormalised for fast reads
    date             = Column(Date,       nullable=False)
    time             = Column(Time,       nullable=False)         # check-in time
    check_out        = Column(Time,       nullable=True)          # last seen time today
    confidence_score = Column(Float,      nullable=False, default=0.0)
    status           = Column(String(20), nullable=False, default="present")
    notes            = Column(Text,       nullable=True)

    # ── Duplicate prevention at the database level ────────────────────────────
    __table_args__ = (
        UniqueConstraint("employee_id", "date", name="uq_employee_date"),
        Index("idx_attendance_date",        "date"),
        Index("idx_attendance_employee_id", "employee_id"),
    )

    employee = relationship("Employee", back_populates="attendance_records")

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "employee_id":      self.employee_id,
            "name":             self.name,
            "date":             str(self.date),
            "time":             str(self.time),
            "check_out":        str(self.check_out) if self.check_out else None,
            "confidence_score": round(self.confidence_score, 4),
            "status":           self.status,
            "notes":            self.notes,
        }

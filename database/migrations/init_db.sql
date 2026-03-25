-- Migration: Initial schema
-- Run automatically by SQLAlchemy on first startup (via models.py Base.metadata.create_all)
-- This file serves as a human-readable reference for the schema.

CREATE TABLE IF NOT EXISTS employees (
    employee_id TEXT    PRIMARY KEY,
    name        TEXT    NOT NULL,
    department  TEXT    NOT NULL,
    email       TEXT    NOT NULL UNIQUE,
    is_active   INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS attendance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT    NOT NULL REFERENCES employees(employee_id) ON DELETE CASCADE,
    date        TEXT    NOT NULL,
    check_in    TEXT    NOT NULL,
    check_out   TEXT,
    confidence  REAL    NOT NULL DEFAULT 0.0,
    status      TEXT    NOT NULL DEFAULT 'present',
    notes       TEXT,
    UNIQUE(employee_id, date)
);

CREATE INDEX IF NOT EXISTS idx_attendance_date        ON attendance(date);
CREATE INDEX IF NOT EXISTS idx_attendance_employee_id ON attendance(employee_id);

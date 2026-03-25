import tempfile
from pathlib import Path

import pytest

from database.db_handler import DBHandler


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield DBHandler(db_path=Path(tmpdir) / "test.db")


def test_add_and_get_employee(db):
    db.add_employee("EMP001", "Alice Smith", "Engineering", "alice@test.com")
    emp = db.get_employee("EMP001")
    assert emp is not None
    assert emp["name"] == "Alice Smith"
    assert emp["department"] == "Engineering"


def test_employee_exists(db):
    db.add_employee("EMP002", "Bob Jones", "HR", "bob@test.com")
    assert db.employee_exists("EMP002") is True
    assert db.employee_exists("EMP999") is False


def test_get_all_employees(db):
    db.add_employee("EMP003", "Carol White", "Sales", "carol@test.com")
    db.add_employee("EMP004", "Dave Brown", "Engineering", "dave@test.com")
    employees = db.get_all_employees()
    assert len(employees) == 2


def test_update_employee(db):
    db.add_employee("EMP005", "Eve Green", "Marketing", "eve@test.com")
    db.update_employee("EMP005", department="Product")
    emp = db.get_employee("EMP005")
    assert emp["department"] == "Product"


def test_delete_employee(db):
    db.add_employee("EMP006", "Frank Blue", "Finance", "frank@test.com")
    result = db.delete_employee("EMP006")
    assert result is True
    assert db.employee_exists("EMP006") is False


def test_mark_attendance(db):
    db.add_employee("EMP007", "Grace Red", "Engineering", "grace@test.com")
    record = db.mark_attendance("EMP007", confidence=0.92)
    assert record is not None
    assert record["employee_id"] == "EMP007"


def test_get_attendance_by_date(db):
    from datetime import date
    db.add_employee("EMP008", "Hank Yellow", "QA", "hank@test.com")
    db.mark_attendance("EMP008", confidence=0.88)
    records = db.get_attendance_by_date(date.today())
    assert len(records) == 1
    assert records[0]["name"] == "Hank Yellow"

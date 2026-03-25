import re
from pathlib import Path

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_employee_id(emp_id: str) -> bool:
    """Employee ID must be alphanumeric, 3–20 characters."""
    return bool(re.fullmatch(r"[A-Za-z0-9_\-]{3,20}", emp_id))


def validate_name(name: str) -> bool:
    """Name must be 2–60 characters, letters and spaces only."""
    return bool(re.fullmatch(r"[A-Za-z\s\-']{2,60}", name.strip()))


def validate_email(email: str) -> bool:
    pattern = r"^[\w\.\+\-]+@[\w\-]+\.[a-zA-Z]{2,}$"
    return bool(re.fullmatch(pattern, email))


def validate_image_extension(path: str | Path) -> bool:
    return Path(path).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS


def validate_department(department: str) -> bool:
    return 1 <= len(department.strip()) <= 50

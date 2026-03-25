"""
main.py — Single entry point for the entire system.

Run:
    python main.py           ← Streamlit UI  (default)
    python main.py --webcam  ← Standalone OpenCV window (no browser)
    python main.py --check   ← Verify environment only
"""
from __future__ import annotations

import sys
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _check_environment() -> bool:
    """Verify Python version, required packages, and project structure."""
    ok = True

    # Python version
    if sys.version_info < (3, 9):
        print(f"[FAIL] Python 3.9+ required — found {sys.version}")
        ok = False
    else:
        print(f"[ OK ] Python {sys.version_info.major}.{sys.version_info.minor}")

    # Required packages
    packages = {
        "cv2":        "opencv-python",
        "numpy":      "numpy",
        "deepface":   "deepface",
        "tensorflow": "tensorflow",
        "streamlit":  "streamlit",
        "sqlalchemy": "SQLAlchemy",
        "pandas":     "pandas",
    }
    for module, pip_name in packages.items():
        try:
            __import__(module)
            print(f"[ OK ] {pip_name}")
        except ImportError:
            print(f"[FAIL] {pip_name} not installed — run: pip install {pip_name}")
            ok = False

    # Critical project files
    critical = [
        ROOT / "config" / "settings.yaml",
        ROOT / "database" / "db_handler.py",
        ROOT / "backend" / "face_detection.py",
        ROOT / "backend" / "face_recognition.py",
        ROOT / "backend" / "registration.py",
        ROOT / "app" / "main_app.py",
    ]
    for path in critical:
        if path.exists():
            print(f"[ OK ] {path.relative_to(ROOT)}")
        else:
            print(f"[FAIL] Missing: {path.relative_to(ROOT)}")
            ok = False

    return ok


def _init_database() -> None:
    """Create all DB tables if they don't exist yet."""
    sys.path.insert(0, str(ROOT))
    from database.db_handler import get_db
    db = get_db()
    db.init_db()
    print("[ OK ] Database initialised")


def _run_streamlit() -> None:
    app_path = ROOT / "app" / "main_app.py"
    print(f"\nStarting Streamlit → http://localhost:8501\n")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path),
         "--server.headless", "false",
         "--browser.gatherUsageStats", "false"],
        check=True,
    )


def _run_webcam() -> None:
    sys.path.insert(0, str(ROOT))
    from backend.webcam import main as webcam_main
    print("Starting OpenCV webcam window — press Q to quit\n")
    webcam_main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Face Employee Recognition System")
    parser.add_argument("--webcam", action="store_true", help="Run standalone OpenCV mode")
    parser.add_argument("--check",  action="store_true", help="Check environment only")
    args = parser.parse_args()

    print("=" * 52)
    print("  Multi-Face Employee Recognition System")
    print("=" * 52)

    if args.check:
        ok = _check_environment()
        sys.exit(0 if ok else 1)

    # Always init DB first (safe to call multiple times — no-op if tables exist)
    sys.path.insert(0, str(ROOT))
    _init_database()

    if args.webcam:
        _run_webcam()
    else:
        _run_streamlit()


if __name__ == "__main__":
    main()

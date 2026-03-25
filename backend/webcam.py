"""
webcam.py — Real-time OpenCV webcam face recognition system.

3-Thread Architecture (display never waits for recognition)
────────────────────────────────────────────────────────────
  Thread 1 — CaptureThread
      Continuously reads raw frames from the camera.
      Stores only the LATEST frame (old ones discarded immediately).

  Thread 2 — RecognitionThread
      Grabs the latest frame whenever the previous recognition pass finishes.
      Runs face detection + DeepFace embedding + cosine match.
      Updates _latest_results (protected by a Lock).
      This thread runs as fast as CPU allows (~2–5 fps on CPU, ~15+ on GPU).

  Main Thread — Display
      Reads the latest frame + latest results every iteration.
      Draws bounding boxes and HUD.
      Calls cv2.imshow() + cv2.waitKey(1) at full camera FPS (30 fps).
      NEVER blocked by recognition — always smooth.

Why it was slow before
──────────────────────
  DeepFace.represent() takes 100–500 ms per face on CPU.
  Running it in the main loop → display updated only 2–5× per second.
  Now recognition is fully off the display path.

Attendance deduplication (two layers)
──────────────────────────────────────
  1. _session_marked set  : in-memory, per-process-run, never hits the DB twice.
  2. DB UNIQUE constraint : catches any edge-case concurrent write.

Keyboard controls
─────────────────
  Q / Esc  quit
  R        reload embeddings from disk (after registering a new employee)
  S        save screenshot to data/temp/
  P        pause / resume recognition (display keeps running)

Run:
    python -m backend.webcam
"""
from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from backend.pipeline import FaceRecognitionPipeline
from backend.face_recognition import RecognitionResult
from database.db_handler import get_db
from utils.config import get_config
from utils.logger import get_logger

log = get_logger(__name__)

# ── Visual constants ──────────────────────────────────────────────────────────
_GREEN  = (0,   210,  80)
_RED    = (0,    50, 220)
_YELLOW = (0,   210, 220)
_WHITE  = (255, 255, 255)
_BLACK  = (0,     0,   0)
_ALPHA  = 0.50
_FONT   = cv2.FONT_HERSHEY_DUPLEX
_FONTSM = cv2.FONT_HERSHEY_SIMPLEX


class WebcamRecognitionSystem:
    """
    Real-time multi-face recognition with automatic attendance marking.

    Parameters
    ----------
    camera_index     : OpenCV device index (0 = default webcam).
    width, height    : Requested capture resolution.
    recognition_size : Resolution used internally by the pipeline.
                       Smaller = faster.  Bboxes are scaled back up.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        recognition_size: tuple[int, int] = (320, 240),
    ):
        self._cfg       = get_config()
        self._db        = get_db()
        # skip_frames=0 here — the RecognitionThread already runs at its own
        # natural rate without blocking the display.
        self._pipeline  = FaceRecognitionPipeline(skip_frames=0)
        self._cam_index = camera_index
        self._disp_w    = width
        self._disp_h    = height
        self._rec_size  = recognition_size
        self._paused    = False
        self._running   = False

        # Attendance: once per session
        self._session_marked: set[str] = set()

        # ── Shared state (protected by locks) ─────────────────────────────
        self._frame_lock   = threading.Lock()
        self._latest_frame: np.ndarray | None = None

        self._result_lock    = threading.Lock()
        self._latest_results: list[RecognitionResult] = []
        # Scale factors so bboxes drawn on display-res frame are correct
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0

        # ── FPS counter (display-side, rolling 60 frames) ─────────────────
        self._frame_times: deque[float] = deque(maxlen=60)
        # Recognition-side FPS
        self._recog_times: deque[float] = deque(maxlen=10)

        self._cap: cv2.VideoCapture | None = None

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Open camera, pre-warm model, start threads, enter display loop."""
        self._open_camera()
        self._prewarm_model()
        self._running = True

        threads = [
            threading.Thread(target=self._capture_loop,   daemon=True, name="CaptureThread"),
            threading.Thread(target=self._recognition_loop, daemon=True, name="RecognitionThread"),
        ]
        for t in threads:
            t.start()

        title = "Face Recognition  |  Q=quit  R=reload  S=screenshot  P=pause"
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, self._disp_w, self._disp_h)

        try:
            self._display_loop(title)
        finally:
            self._running = False
            for t in threads:
                t.join(timeout=2)
            if self._cap:
                self._cap.release()
            cv2.destroyAllWindows()
            log.info("System stopped. Attendance marked: {}", len(self._session_marked))

    # ── Thread 1 — Camera capture ─────────────────────────────────────────────

    def _capture_loop(self) -> None:
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._frame_lock:
                self._latest_frame = frame

    # ── Thread 2 — Recognition ────────────────────────────────────────────────

    def _recognition_loop(self) -> None:
        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            with self._frame_lock:
                frame = self._latest_frame

            if frame is None:
                time.sleep(0.01)
                continue

            # Downscale for fast inference
            small = cv2.resize(frame, self._rec_size)
            disp_h, disp_w = frame.shape[:2]
            sx = disp_w / self._rec_size[0]
            sy = disp_h / self._rec_size[1]

            results = self._pipeline.process_frame(small)
            scaled  = _scale_bboxes(results, sx, sy)

            # Mark attendance for newly recognised employees
            for r in scaled:
                if r.matched:
                    self._try_mark_attendance(r)

            with self._result_lock:
                self._latest_results = scaled
                self._scale_x = sx
                self._scale_y = sy

            self._recog_times.append(time.time())

    # ── Main thread — Display ─────────────────────────────────────────────────

    def _display_loop(self, window: str) -> None:
        while True:
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
                continue

            display = frame.copy()

            with self._result_lock:
                results = list(self._latest_results)

            self._draw_faces(display, results)
            self._draw_hud(display)

            self._frame_times.append(time.time())
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("r"), ord("R")):
                self._pipeline.reload_embeddings()
                log.info("Embeddings reloaded")
            elif key in (ord("s"), ord("S")):
                self._screenshot(display)
            elif key in (ord("p"), ord("P")):
                self._paused = not self._paused
                log.info("Recognition {}", "paused" if self._paused else "resumed")

    # ── Attendance ────────────────────────────────────────────────────────────

    def _try_mark_attendance(self, r: RecognitionResult) -> None:
        if r.employee_id in self._session_marked:
            return
        try:
            self._db.mark_attendance(r.employee_id, r.name, r.confidence)
            self._session_marked.add(r.employee_id)
            log.info("Attendance ✓  {} ({})  {:.1%}", r.name, r.employee_id, r.confidence)
        except Exception as e:
            log.error("Attendance write failed for '{}': {}", r.employee_id, e)

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _draw_faces(
        self, frame: np.ndarray, results: list[RecognitionResult]
    ) -> None:
        for r in results:
            if r.bbox is None:
                continue
            x, y, w, h = r.bbox
            color    = _GREEN if r.matched else _RED
            tag      = "Employee" if r.matched else "Unknown"
            marked   = r.employee_id in self._session_marked
            conf_str = f"{r.confidence:.1%}"

            # Bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Corner accent lines (cosmetic)
            cl = 18
            for (px, py, dx, dy) in [
                (x, y, 1, 1), (x+w, y, -1, 1), (x, y+h, 1, -1), (x+w, y+h, -1, -1)
            ]:
                cv2.line(frame, (px, py), (px + dx*cl, py), color, 3)
                cv2.line(frame, (px, py), (px, py + dy*cl), color, 3)

            # Name label
            name_text = r.name
            _pill(frame, name_text,  x, y - 32, color, scale=0.62, thick=2)

            # Tag + confidence + ✓
            sub = f"{tag}  {conf_str}" + ("  \u2713" if marked else "")
            _pill(frame, sub, x, y - 10, color, scale=0.50, thick=1)

    def _draw_hud(self, frame: np.ndarray) -> None:
        disp_fps  = self._fps(self._frame_times)
        recog_fps = self._fps(self._recog_times)
        h = frame.shape[0]

        # Top-left panel
        lines = [
            (f"Display  {disp_fps:5.1f} fps",  _GREEN),
            (f"Recog    {recog_fps:5.1f} fps",  _YELLOW if self._paused else _WHITE),
            (f"Marked   {len(self._session_marked)}",        _WHITE),
        ]
        if self._paused:
            lines.insert(0, ("PAUSED", _YELLOW))

        panel_w, lh = 210, 22
        overlay = frame.copy()
        cv2.rectangle(overlay, (6, 6), (6 + panel_w, 6 + len(lines) * lh + 8), _BLACK, -1)
        cv2.addWeighted(overlay, _ALPHA, frame, 1 - _ALPHA, 0, frame)

        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (12, 6 + (i + 1) * lh),
                        _FONTSM, 0.52, color, 1, cv2.LINE_AA)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _open_camera(self) -> None:
        self._cap = cv2.VideoCapture(self._cam_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # fallback without backend hint
            self._cap = cv2.VideoCapture(self._cam_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._cam_index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._disp_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._disp_h)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # drop stale frames
        log.info(
            "Camera {}  →  {}x{}",
            self._cam_index,
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def _prewarm_model(self) -> None:
        """
        Run one dummy inference so DeepFace downloads + compiles the model
        before the first real frame arrives. Without this the first recognition
        call blocks for 5–30 s while weights are loaded.
        """
        log.info("Pre-warming DeepFace model (Facenet512) — first run may download weights…")
        dummy = np.zeros((160, 160, 3), dtype=np.uint8)
        try:
            from deepface import DeepFace
            DeepFace.represent(
                img_path=dummy,
                model_name=self._cfg.recognition_model,
                detector_backend="skip",
                enforce_detection=False,
            )
            log.info("Model ready")
        except Exception as e:
            log.warning("Pre-warm failed ({}), will load on first frame", e)

    @staticmethod
    def _fps(times: deque) -> float:
        if len(times) < 2:
            return 0.0
        elapsed = times[-1] - times[0]
        return (len(times) - 1) / elapsed if elapsed > 0 else 0.0

    def _screenshot(self, frame: np.ndarray) -> None:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self._cfg.temp_dir / f"screenshot_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        log.info("Screenshot → {}", path)


# ── Module helpers ────────────────────────────────────────────────────────────

def _scale_bboxes(
    results: list[RecognitionResult], sx: float, sy: float
) -> list[RecognitionResult]:
    out = []
    for r in results:
        bbox = (int(r.bbox[0]*sx), int(r.bbox[1]*sy),
                int(r.bbox[2]*sx), int(r.bbox[3]*sy)) if r.bbox else None
        out.append(RecognitionResult(
            employee_id=r.employee_id, name=r.name,
            confidence=r.confidence, matched=r.matched, bbox=bbox,
        ))
    return out


def _pill(
    img: np.ndarray, text: str,
    x: int, y: int, color: tuple,
    scale: float = 0.6, thick: int = 1,
) -> None:
    """Draw text with a semi-transparent dark background for readability."""
    (tw, th), bl = cv2.getTextSize(text, _FONT, scale, thick)
    pad = 4
    x1, y1 = max(0, x - pad), max(0, y - th - pad)
    x2, y2 = min(img.shape[1], x + tw + pad), min(img.shape[0], y + bl + pad)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), _BLACK, -1)
    cv2.addWeighted(overlay, _ALPHA, img, 1 - _ALPHA, 0, img)
    cv2.putText(img, text, (x, y), _FONT, scale, color, thick, cv2.LINE_AA)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    cfg = get_config()
    WebcamRecognitionSystem(
        camera_index     = cfg.webcam_device,
        width            = cfg.webcam_resolution[0],
        height           = cfg.webcam_resolution[1],
        recognition_size = (320, 240),
    ).run()


if __name__ == "__main__":
    main()

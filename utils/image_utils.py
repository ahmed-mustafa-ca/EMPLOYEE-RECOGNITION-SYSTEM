from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils.logger import get_logger

log = get_logger(__name__)


def read_image(path: str | Path) -> np.ndarray:
    """Read an image from disk and return as BGR numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def save_image(img: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def resize_image(img: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def ndarray_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr_to_rgb(img))


def pil_to_ndarray(img: Image.Image) -> np.ndarray:
    return rgb_to_bgr(np.array(img))


def crop_face(img: np.ndarray, x: int, y: int, w: int, h: int, padding: int = 20) -> np.ndarray:
    """Crop a face region with optional padding."""
    h_img, w_img = img.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w_img, x + w + padding)
    y2 = min(h_img, y + h + padding)
    return img[y1:y2, x1:x2]


def draw_face_box(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str = "",
    confidence: float = 0.0,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw a bounding box with label on a frame."""
    img = img.copy()
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    if label:
        text = f"{label} ({confidence:.1%})" if confidence else label
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def is_valid_image(path: str | Path) -> bool:
    try:
        img = cv2.imread(str(path))
        return img is not None
    except Exception:
        return False

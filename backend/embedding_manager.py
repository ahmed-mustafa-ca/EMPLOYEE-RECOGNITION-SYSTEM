import pickle
from pathlib import Path

import numpy as np

from utils.config import get_config
from utils.logger import get_logger

log = get_logger(__name__)


class EmbeddingManager:
    """
    Persists and retrieves face embeddings as .pkl files.
    One file per employee: embeddings/<employee_id>.pkl
    Each file stores a list of embedding arrays (multiple photos per person).
    """

    def __init__(self):
        self._cfg = get_config()
        self._dir = self._cfg.embeddings_dir

    def _path(self, employee_id: str) -> Path:
        return self._dir / f"{employee_id}.pkl"

    def save(self, employee_id: str, name: str, embeddings: list[np.ndarray]) -> None:
        """Persist embeddings for one employee (overwrites existing)."""
        payload = {"employee_id": employee_id, "name": name, "embeddings": embeddings}
        with open(self._path(employee_id), "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Saved {} embedding(s) for employee '{}'", len(embeddings), employee_id)

    def append(self, employee_id: str, name: str, embedding: np.ndarray) -> None:
        """Add a single new embedding without replacing existing ones."""
        existing = self._load_one(employee_id)
        if existing:
            embeddings = existing["embeddings"] + [embedding]
        else:
            embeddings = [embedding]
        self.save(employee_id, name, embeddings)

    def _load_one(self, employee_id: str) -> dict | None:
        p = self._path(employee_id)
        if not p.exists():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def load_all(self) -> list[tuple[str, str, np.ndarray]]:
        """
        Returns a flat list of (employee_id, name, embedding) tuples.
        Multiple embeddings per employee are all included for best-match search.
        """
        results = []
        for pkl_file in self._dir.glob("*.pkl"):
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                for emb in data["embeddings"]:
                    results.append((data["employee_id"], data["name"], emb))
            except Exception as e:
                log.error("Failed to load embedding file {}: {}", pkl_file.name, e)
        log.debug("Loaded {} embedding vector(s) from disk", len(results))
        return results

    def delete(self, employee_id: str) -> bool:
        p = self._path(employee_id)
        if p.exists():
            p.unlink()
            log.info("Deleted embeddings for employee '{}'", employee_id)
            return True
        return False

    def exists(self, employee_id: str) -> bool:
        return self._path(employee_id).exists()

    def list_employees(self) -> list[str]:
        return [p.stem for p in self._dir.glob("*.pkl")]

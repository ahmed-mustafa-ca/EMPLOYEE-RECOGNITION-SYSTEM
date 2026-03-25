from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.face_recognition import FaceRecognizer, RecognitionResult, UNKNOWN
from backend.embedding_manager import EmbeddingManager


@pytest.fixture
def mock_emb_mgr():
    mgr = MagicMock(spec=EmbeddingManager)
    mgr.load_all.return_value = []
    return mgr


@pytest.fixture
def recognizer(mock_emb_mgr):
    return FaceRecognizer(mock_emb_mgr)


@pytest.fixture
def dummy_frame():
    return np.zeros((160, 160, 3), dtype=np.uint8)


def test_recognize_returns_unknown_when_no_embeddings(recognizer, dummy_frame, mock_emb_mgr):
    mock_emb_mgr.load_all.return_value = []
    with patch.object(recognizer, "get_embedding", return_value=np.random.rand(512).astype(np.float32)):
        result = recognizer.recognize(dummy_frame)
    assert result.matched is False
    assert result.name == "Unknown"


def test_recognize_returns_unknown_when_embedding_fails(recognizer, dummy_frame):
    with patch.object(recognizer, "get_embedding", return_value=None):
        result = recognizer.recognize(dummy_frame)
    assert result == UNKNOWN


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sim = FaceRecognizer._cosine_similarity(v, v)
    assert abs(sim - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    sim = FaceRecognizer._cosine_similarity(a, b)
    assert abs(sim - 0.0) < 1e-6


def test_cosine_similarity_zero_vector():
    a = np.zeros(3, dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sim = FaceRecognizer._cosine_similarity(a, b)
    assert sim == 0.0


def test_recognize_batch_returns_list(recognizer, dummy_frame):
    with patch.object(recognizer, "get_embedding", return_value=None):
        results = recognizer.recognize_batch([dummy_frame, dummy_frame])
    assert isinstance(results, list)
    assert len(results) == 2

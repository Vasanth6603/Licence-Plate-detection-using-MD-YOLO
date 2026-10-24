"""Unit tests for preprocessing."""

import numpy as np
import pytest

from core.preprocessing import PreprocessingError, letterbox, load_image, preprocess


def test_preprocess_returns_batched_tensor():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    tensor = preprocess(image, img_size=640, half=False)
    assert tensor.ndim == 4  # NCHW
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 3
    assert tensor.shape[2] == 640
    assert tensor.shape[3] == 640
    assert tensor.dtype == np.float32


def test_letterbox_preserves_aspect():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    img, gain, pad = letterbox(image, new_shape=(640, 640), auto=False, scaleup=False)
    assert img.shape[:2] == (640, 640)
    assert gain <= 1.0


def test_load_image_bytes_invalid():
    try:
        load_image(b"not an image")
        pytest.fail("Expected PreprocessingError")
    except PreprocessingError:
        pass

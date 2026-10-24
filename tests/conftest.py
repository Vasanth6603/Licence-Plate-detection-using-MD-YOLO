"""Pytest configuration and shared fixtures."""

import numpy as np
import pytest

from core.config import Settings


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def blank_image():
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection():
    from core.postprocessing import Detection

    return Detection(
        x1=100.0,
        y1=100.0,
        x2=200.0,
        y2=200.0,
        confidence=0.9,
        class_id=0,
        class_name="licence",
    )

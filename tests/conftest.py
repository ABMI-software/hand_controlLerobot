"""
Shared pytest fixtures for all tests.
"""
import pytest
import cv2
import numpy as np

@pytest.fixture
def mock_camera_frame():
    """Create a mock camera frame for testing."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)

@pytest.fixture
def mock_object_detection():
    """Create mock object detection results."""
    return [
        {
            'class': 0,
            'confidence': 0.95,
            'bbox': [100, 100, 200, 200]
        }
    ]
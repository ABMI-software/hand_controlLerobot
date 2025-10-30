#!/usr/bin/env python3
"""
Test script for error handling and edge cases.
Tests system behavior under various error conditions.
"""

import cv2
import numpy as np
import time
from pathlib import Path
import unittest
from hand_teleop.cameras.camera_manager import CameraManager
from hand_teleop.detection.object_detector import ObjectDetector
from hand_teleop.tracking.tracker import HandTracker

class TestErrorHandling(unittest.TestCase):
    def setUp(self):
        self.cam_manager = CameraManager()
        self.model_path = "yolov8n.pt"

    def test_invalid_camera_id(self):
        """Test handling of invalid camera ID."""
        print("\nTesting invalid camera ID...")
        with self.assertRaises(Exception):
            self.cam_manager.add_camera("invalid", 999)
        print("✓ Invalid camera ID handled correctly")

    def test_duplicate_camera_name(self):
        """Test handling of duplicate camera names."""
        print("\nTesting duplicate camera name...")
        self.cam_manager.add_camera("test", 0)
        with self.assertRaises(ValueError):
            self.cam_manager.add_camera("test", 1)
        print("✓ Duplicate camera name handled correctly")

    def test_missing_model_file(self):
        """Test handling of missing YOLOv8 model file."""
        print("\nTesting missing model file...")
        with self.assertRaises(Exception):
            ObjectDetector("nonexistent_model.pt")
        print("✓ Missing model file handled correctly")

    def test_camera_disconnection(self):
        """Test handling of camera disconnection."""
        print("\nTesting camera disconnection handling...")
        self.cam_manager.add_camera("test", 0)
        self.cam_manager.start()
        
        # Get initial frame
        frame, _ = self.cam_manager.get_frame("test")
        self.assertIsNotNone(frame, "Camera should provide initial frame")
        
        # Simulate disconnection by stopping camera
        self.cam_manager.cameras["test"].stop()
        
        # Check handling of disconnected camera
        frame, _ = self.cam_manager.get_frame("test")
        self.assertIsNone(frame, "Disconnected camera should return None")
        print("✓ Camera disconnection handled correctly")

    def test_invalid_frame_processing(self):
        """Test handling of invalid frames in object detection."""
        print("\nTesting invalid frame processing...")
        if Path(self.model_path).exists():
            detector = ObjectDetector(self.model_path)
            
            # Test with None frame
            with self.assertRaises(Exception):
                detector.detect(None)
            
            # Test with empty frame
            with self.assertRaises(Exception):
                detector.detect(np.array([]))
            
            # Test with invalid shape
            with self.assertRaises(Exception):
                detector.detect(np.zeros((10, 10)))
            
            print("✓ Invalid frame processing handled correctly")
        else:
            print("⚠ Skipping invalid frame test (model file not found)")

    def test_resource_cleanup(self):
        """Test proper resource cleanup."""
        print("\nTesting resource cleanup...")
        
        # Initialize resources
        self.cam_manager.add_camera("test", 0)
        self.cam_manager.start()
        
        # Test cleanup
        self.cam_manager.stop()
        
        # Verify cleanup
        self.assertFalse(self.cam_manager.is_running)
        for camera in self.cam_manager.cameras.values():
            self.assertFalse(camera.is_running)
            self.assertIsNone(camera.cap)
        
        print("✓ Resource cleanup handled correctly")

    def tearDown(self):
        self.cam_manager.stop()

def run_error_handling_tests():
    print("Running error handling and edge case tests...")
    unittest.main(argv=[''], verbosity=2, exit=False)

if __name__ == "__main__":
    run_error_handling_tests()
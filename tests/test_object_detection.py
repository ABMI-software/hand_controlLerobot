#!/usr/bin/env python3
"""
Test script for the object detection system.
Tests YOLOv8 initialization, object detection, and visualization.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from hand_teleop.cameras.camera_manager import CameraManager
from hand_teleop.detection.object_detector import ObjectDetector

def test_object_detection():
    print("Testing object detection system...")
    
    # Check if YOLOv8 model exists
    model_path = "yolov8n.pt"
    if not Path(model_path).exists():
        print(f"✗ YOLOv8 model not found at {model_path}")
        print("Please download it using:")
        print("wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt")
        return

    # Initialize camera
    print("\n1. Setting up camera...")
    cam_manager = CameraManager()
    try:
        cam_manager.add_camera("arducam", 0, width=1280, height=720)
        cam_manager.start()
        print("✓ Camera initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize camera: {e}")
        return

    # Initialize object detector
    print("\n2. Loading YOLOv8 model...")
    try:
        detector = ObjectDetector(model_path, conf_threshold=0.5)
        print("✓ Object detector initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize object detector: {e}")
        return

    # Test detection
    print("\n3. Testing object detection...")
    print("Running detection for 30 seconds. Press 'q' to quit early.")
    print("Move different objects in front of the camera.")
    
    start_time = time.time()
    frame_count = 0
    detections_count = 0
    
    while time.time() - start_time < 30:
        # Get frame
        frame, _ = cam_manager.get_frame("arducam")
        if frame is None:
            continue
            
        frame_count += 1
        
        # Detect objects
        try:
            detections = detector.detect(frame)
            detections_count += len(detections)
            
            # Visualize detections
            viz_frame = detector.visualize(frame, detections)
            
            # Add FPS and detection count
            fps = frame_count / (time.time() - start_time)
            avg_detections = detections_count / frame_count
            
            cv2.putText(viz_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(viz_frame, f"Detections: {len(detections)}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(viz_frame, f"Avg Detections: {avg_detections:.1f}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Object Detection", viz_frame)
            
        except Exception as e:
            print(f"✗ Detection error: {e}")
            break
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cam_manager.stop()
    cv2.destroyAllWindows()

    # Print results
    print("\n=== Test Results ===")
    print(f"Frames processed: {frame_count}")
    print(f"Average FPS: {frame_count / 30:.1f}")
    print(f"Total detections: {detections_count}")
    print(f"Average detections per frame: {detections_count / frame_count:.1f}")
    print(f"Status: {'✓ OK' if frame_count > 0 and detections_count > 0 else '✗ Failed'}")

if __name__ == "__main__":
    test_object_detection()
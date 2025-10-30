#!/usr/bin/env python3
"""
Test script for object detection using Arducam.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from hand_teleop.cameras.camera_manager import CameraManager
from hand_teleop.detection.object_detector import ObjectDetector

def test_arducam_detection():
    print("Testing object detection with Arducam...")
    
    # Initialize camera manager
    print("\n1. Setting up Arducam...")
    cam_manager = CameraManager()
    try:
        cam_manager.add_camera(
            name="arducam",
            camera_id=0,  # /dev/video0
            width=1280,
            height=720,
            fps=30
        )
        cam_manager.start()
        print("✓ Arducam initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize Arducam: {e}")
        return False

    # Initialize object detector
    print("\n2. Loading YOLOv8 model...")
    try:
        model_path = "yolov8n.pt"
        if not Path(model_path).exists():
            print(f"Downloading YOLOv8 model to {model_path}")
            import torch
            torch.hub.download_url_to_file(
                'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
                model_path
            )
            
        detector = ObjectDetector(model_path, conf_threshold=0.5)
        print("✓ Object detector initialized")
    except Exception as e:
        print(f"✗ Failed to initialize object detector: {e}")
        cam_manager.stop()
        return False

    # Test detection
    print("\n3. Running object detection...")
    print("Press 'q' to quit, 's' to save a frame")
    
    start_time = time.time()
    frame_count = 0
    detection_count = 0
    saved_frame = False
    
    while True:
        frame, _ = cam_manager.get_frame("arducam")
        if frame is None:
            continue
            
        frame_count += 1
        
        # Run detection
        detections = detector.detect(frame)
        detection_count += len(detections)
        
        # Visualize detections
        viz_frame = detector.visualize(frame, detections)
        
        # Add stats
        fps = frame_count / (time.time() - start_time)
        avg_detections = detection_count / frame_count if frame_count > 0 else 0
        cv2.putText(viz_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(viz_frame, f"Detections: {avg_detections:.1f}/frame", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Arducam Object Detection", viz_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and not saved_frame:
            filename = f"arducam_detection_{int(time.time())}.jpg"
            cv2.imwrite(filename, viz_frame)
            print(f"\nSaved frame to {filename}")
            saved_frame = True

    # Print stats
    print("\nTest Results:")
    print(f"Total frames: {frame_count}")
    print(f"Average FPS: {frame_count / (time.time() - start_time):.1f}")
    print(f"Average detections per frame: {detection_count / frame_count:.1f}")
    
    # Cleanup
    cam_manager.stop()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    if test_arducam_detection():
        print("\nObject detection test completed successfully!")
    else:
        print("\nObject detection test failed!")
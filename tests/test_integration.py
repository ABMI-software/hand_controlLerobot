#!/usr/bin/env python3
"""
Test script for the mode switching and integration.
Tests mode transitions and integration between components.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from hand_teleop.cameras.camera_manager import CameraManager
from hand_teleop.detection.object_detector import ObjectDetector
from hand_teleop.tracking.tracker import HandTracker

def test_integration():
    print("Testing system integration and mode switching...")
    
    # Initialize components
    print("\n1. Initializing components...")
    
    # Camera system
    try:
        cam_manager = CameraManager()
        cam_manager.add_camera("arducam", 0, width=1280, height=720)
        cam_manager.add_camera("webcam", 1, width=640, height=480)
        cam_manager.start()
        print("✓ Camera system initialized")
    except Exception as e:
        print(f"✗ Camera system initialization failed: {e}")
        return

    # Object detector
    try:
        model_path = "yolov8n.pt"
        if not Path(model_path).exists():
            raise FileNotFoundError(f"YOLOv8 model not found at {model_path}")
        detector = ObjectDetector(model_path, conf_threshold=0.5)
        print("✓ Object detector initialized")
    except Exception as e:
        print(f"✗ Object detector initialization failed: {e}")
        return

    # Hand tracker
    try:
        hand_tracker = HandTracker(cam_idx=1, model="wilor")
        print("✓ Hand tracker initialized")
    except Exception as e:
        print(f"✗ Hand tracker initialization failed: {e}")
        return

    # Test mode switching
    print("\n2. Testing mode switching...")
    print("Controls:")
    print("  'd' - Detection mode")
    print("  't' - Teleoperation mode")
    print("  'a' - Auto-pick mode")
    print("  'q' - Quit")

    current_mode = "detection"
    start_time = time.time()
    frame_count = 0
    mode_switches = 0

    while time.time() - start_time < 60:  # Run for 1 minute
        frame_count += 1
        
        # Get frames from both cameras
        arducam_frame, _ = cam_manager.get_frame("arducam")
        webcam_frame, _ = cam_manager.get_frame("webcam")
        
        if arducam_frame is None or webcam_frame is None:
            continue

        # Process frames based on current mode
        if current_mode == "detection":
            # Object detection mode
            detections = detector.detect(arducam_frame)
            viz_frame = detector.visualize(arducam_frame, detections)
            cv2.putText(viz_frame, "Detection Mode", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Main View", viz_frame)
            
        elif current_mode == "teleoperation":
            # Teleoperation mode
            hand_pose = hand_tracker.track(webcam_frame)
            # Combine both views
            combined = np.hstack((arducam_frame, webcam_frame))
            cv2.putText(combined, "Teleoperation Mode", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Main View", combined)
            
        elif current_mode == "auto_pick":
            # Auto-pick mode
            detections = detector.detect(arducam_frame)
            viz_frame = detector.visualize(arducam_frame, detections)
            cv2.putText(viz_frame, "Auto-Pick Mode", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Main View", viz_frame)

        # Handle mode switching
        key = cv2.waitKey(1) & 0xFF
        new_mode = None
        
        if key == ord('d'):
            new_mode = "detection"
        elif key == ord('t'):
            new_mode = "teleoperation"
        elif key == ord('a'):
            new_mode = "auto_pick"
        elif key == ord('q'):
            break

        if new_mode and new_mode != current_mode:
            current_mode = new_mode
            mode_switches += 1
            print(f"\nSwitched to {current_mode} mode")

    # Cleanup
    cam_manager.stop()
    cv2.destroyAllWindows()

    # Print results
    print("\n=== Test Results ===")
    duration = time.time() - start_time
    print(f"Test duration: {duration:.1f} seconds")
    print(f"Frames processed: {frame_count}")
    print(f"Average FPS: {frame_count / duration:.1f}")
    print(f"Mode switches: {mode_switches}")
    print(f"Status: {'✓ OK' if frame_count > 0 and mode_switches > 0 else '✗ Failed'}")

if __name__ == "__main__":
    test_integration()
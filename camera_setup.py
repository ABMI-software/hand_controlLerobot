#!/usr/bin/env python3
"""
Camera setup script for configuring Arducam and Astra depth camera.
"""

import cv2
import numpy as np
import time
from pathlib import Path
import subprocess
import os
from hand_teleop.cameras.camera_manager import CameraManager

def setup_astra():
    """Setup Astra depth camera with OpenNI2."""
    print("Checking OpenNI2 dependencies...")
    
    # Check if OpenNI2 is installed
    if not os.path.exists('/usr/lib/libOpenNI2.so'):
        print("Installing OpenNI2 dependencies...")
        try:
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'libopenni2-0', 'openni2-utils'], check=True)
            subprocess.run(['sudo', 'apt-get', 'install', '-y', 'libopenni2-dev'], check=True)
            print("✓ OpenNI2 dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install OpenNI2: {e}")
            return False
    
    # Test if OpenCV can access the camera
    try:
        cap = cv2.VideoCapture(cv2.CAP_OPENNI2)
        if not cap.isOpened():
            raise RuntimeError("Cannot open Astra camera through OpenNI2")
        cap.release()
        print("✓ Astra camera detected through OpenNI2")
        return True
    except Exception as e:
        print(f"Error accessing Astra camera: {e}")
        return False

def setup_cameras():
    print("Setting up camera system...")
    
    # Initialize camera manager
    cam_manager = CameraManager()
    
    # Setup Arducam (video0) for object detection
    print("\n1. Setting up Arducam...")
    try:
        # Try to open the camera first to verify it works
        test_cap = cv2.VideoCapture(0)
        if not test_cap.isOpened():
            raise RuntimeError("Cannot open Arducam - please check connection")
        test_cap.release()
        
        cam_manager.add_camera(
            name="arducam",
            camera_id=0,  # /dev/video0
            width=1280,
            height=720,
            fps=30
        )
        print("✓ Arducam configured successfully")
    except Exception as e:
        print(f"✗ Failed to configure Arducam: {e}")
        return None
    
    # Setup OpenNI2 for Astra camera
    print("\n2. Setting up Astra depth camera...")
    if not setup_astra():
        print("✗ Failed to initialize OpenNI2 interface for Astra camera")
        return None
        
    try:
        cam_manager.add_camera(
            name="astra",
            camera_id=-1,  # Special ID for OpenNI interface
            width=640,
            height=480,
            fps=30
        )
        print("✓ Astra depth camera configured successfully")
    except Exception as e:
        print(f"✗ Failed to configure Astra: {e}")
        return None

    # Start both cameras
    print("\n3. Starting cameras...")
    try:
        cam_manager.start()
        print("✓ Cameras started successfully")
    except Exception as e:
        print(f"✗ Failed to start cameras: {e}")
        return None

    # Test frame capture from both cameras
    print("\n4. Testing frame capture...")
    print("Capturing frames for 5 seconds. Press 'q' to quit early.")
    
    start_time = time.time()
    frame_counts = {"arducam": 0, "astra": 0}
    
    while time.time() - start_time < 5:
        # Get frames from both cameras
        arducam_frame, _ = cam_manager.get_frame("arducam")
        astra_frame, _ = cam_manager.get_frame("astra")
        
        # Show and count Arducam frames
        if arducam_frame is not None:
            frame_counts["arducam"] += 1
            cv2.putText(arducam_frame, "Arducam Feed", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Arducam (Object Detection)", arducam_frame)
            
        # Show and count Astra frames
        if astra_frame is not None:
            frame_counts["astra"] += 1
            cv2.putText(astra_frame, "Astra Feed", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Astra (Hand Tracking)", astra_frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Print frame statistics
    print("\nFrame capture statistics:")
    for camera, count in frame_counts.items():
        fps = count / 5  # 5 seconds test duration
        print(f"{camera}: {count} frames ({fps:.1f} FPS)")
    
    # Cleanup
    cv2.destroyAllWindows()
    
    return cam_manager

if __name__ == "__main__":
    cam_manager = setup_cameras()
    if cam_manager:
        print("\nCamera setup completed successfully!")
        cam_manager.stop()
    else:
        print("\nCamera setup failed!")
#!/usr/bin/env python3
"""
Test script for the camera system.
Tests camera initialization, frame capture, and synchronization.
"""

import cv2
import numpy as np
import time
import os
from hand_teleop.cameras.camera_manager import CameraManager

def list_available_cameras():
    """List all available video devices in /dev/."""
    video_devices = []
    for device in os.listdir('/dev'):
        if device.startswith('video'):
            path = f'/dev/{device}'
            try:
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    # Try to get camera information
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    video_devices.append({
                        'path': path,
                        'width': width,
                        'height': height
                    })
                cap.release()
            except Exception as e:
                print(f"Could not open {path}: {e}")
    return video_devices

def test_camera_system():
    print("Testing camera system...")
    
    # Initialize camera manager
    cam_manager = CameraManager()
    
    # Test camera addition
    print("\n1. Testing camera initialization...")
    
    # List all available cameras
    print("Scanning for available cameras...")
    video_devices = list_available_cameras()
    
    if not video_devices:
        print("✗ No cameras found!")
        return
    
    print("\nFound cameras:")
    for i, device in enumerate(video_devices):
        print(f"Camera {i}: {device['path']}")
        print(f"  Resolution: {device['width']}x{device['height']}")

    try:
        # Initialize first available camera as Arducam
        print("\nInitializing main camera...")
        arducam_path = video_devices[0]['path']
        print(f"Using {arducam_path} as main camera")
        
        # Extract index from video device path
        arducam_idx = int(arducam_path.replace('/dev/video', ''))
        cam_manager.add_camera("main", arducam_idx, width=1280, height=720)
        print("✓ Main camera added successfully")

    except Exception as e:
        print(f"✗ Failed to add cameras: {e}")
        return

    # Start cameras
    print("\n2. Testing camera start...")
    try:
        cam_manager.start()
        print("✓ Camera started successfully")
    except Exception as e:
        print(f"✗ Failed to start camera: {e}")
        return

    # Test frame capture
    print("\n3. Testing frame capture...")
    print("Capturing frames for 10 seconds. Press 'q' to quit early.")
    print("Press 's' to save a frame.")
    
    start_time = time.time()
    frame_counts = {"main": 0}
    saved_frame = False
    
    while time.time() - start_time < 10:
        frames = cam_manager.get_frames()
        
        # Display and count frames
        for camera_name, (frame, timestamp) in frames.items():
            if frame is not None:
                frame_counts[camera_name] += 1
                
                # Calculate FPS
                elapsed_time = time.time() - start_time
                fps = frame_counts[camera_name] / elapsed_time
                
                # Create info display
                height, width = frame.shape[:2]
                info_text = [
                    f"Camera: {camera_name}",
                    f"FPS: {fps:.1f}",
                    f"Resolution: {width}x{height}",
                    f"Frames: {frame_counts[camera_name]}",
                    "Press 's' to save frame",
                    "Press 'q' to quit"
                ]
                
                # Add info to frame
                for i, text in enumerate(info_text):
                    y = 30 + (i * 30)
                    cv2.putText(frame, text, (10, y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow(camera_name, frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and not saved_frame:
                    # Save frame
                    filename = f"camera_test_{camera_name}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"\nSaved frame to {filename}")
                    saved_frame = True
            else:
                print(f"\rNo frame from {camera_name}", end="", flush=True)

    # Stop cameras
    print("\n4. Testing camera shutdown...")
    try:
        cam_manager.stop()
        cv2.destroyAllWindows()
        print("✓ Cameras stopped successfully")
    except Exception as e:
        print(f"✗ Failed to stop cameras: {e}")
        return

    # Print results
    print("\n=== Test Results ===")
    for camera_name, count in frame_counts.items():
        fps = count / (time.time() - start_time)
        print(f"\nCamera: {camera_name}")
        print(f"  - Total frames: {count}")
        print(f"  - Average FPS: {fps:.1f}")
        print(f"  - Status: {'✓ OK' if fps > 15 else '✗ Low FPS'}")
        if saved_frame:
            print(f"  - Sample frame saved as: camera_test_{camera_name}.jpg")

if __name__ == "__main__":
    test_camera_system()
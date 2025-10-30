#!/usr/bin/env python3
"""
Camera service for OpenNI devices (Astra/Orbbec).
"""
import cv2
import numpy as np
import time
from pathlib import Path
import subprocess
import signal
import sys
import os

def cleanup_handler(signum, frame):
    """Clean up shared memory on exit."""
    print("\nCleaning up...")
    for f in Path("/dev/shm").glob("oni_*"):
        try:
            f.unlink()
        except Exception:
            pass
    sys.exit(0)

def setup_shared_memory(width, height):
    """Set up shared memory files for OpenNI stream."""
    info_file = Path("/dev/shm/oni_info.txt")
    info_file.write_text(f"CW={width}\nCH={height}\n")
    return True

def main():
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    # Clean up any existing shared memory
    for f in Path("/dev/shm").glob("oni_*"):
        try:
            f.unlink()
        except Exception:
            pass
    
    # Initialize OpenNI device
    cap = cv2.VideoCapture(cv2.CAP_OPENNI2)
    if not cap.isOpened():
        print("Failed to open OpenNI2 device")
        sys.exit(1)
    
    # Set up shared memory
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    setup_shared_memory(width, height)
    
    print(f"Started Astra camera service ({width}x{height})")
    tick = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Write frame to shared memory
            if frame is not None:
                frame_path = Path("/dev/shm/oni_color.rgb")
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame_path.write_bytes(frame_bgr.tobytes())
                
                # Update tick
                tick += 1
                Path("/dev/shm/oni_tick.txt").write_text(str(tick))
            
            time.sleep(1/30)  # 30 FPS
            
    except KeyboardInterrupt:
        cleanup_handler(None, None)
    finally:
        cap.release()

if __name__ == "__main__":
    main()
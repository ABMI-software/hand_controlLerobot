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

def main():
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    # Clean up any existing shared memory
    for f in Path("/dev/shm").glob("oni_*"):
        try:
            f.unlink()
        except Exception:
            pass
    
    # Start OpenNI camera service
    try:
        subprocess.run(["oni_grabber", "--no-ir"], check=True)
    except KeyboardInterrupt:
        cleanup_handler(None, None)
    except Exception as e:
        print(f"Failed to start camera: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
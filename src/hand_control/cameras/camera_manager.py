"""
Camera manager for handling multiple cameras.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional

class CameraManager:
    """Manages multiple camera inputs."""
    
    def __init__(self):
        """Initialize an empty camera manager."""
        self.cameras: Dict[str, cv2.VideoCapture] = {}
        self.camera_settings: Dict[str, Dict] = {}
        
    def add_camera(self, name: str, camera_id: int, width: int = 640, height: int = 480, fps: int = 30):
        """Add a new camera to the manager.
        
        Args:
            name: Unique identifier for the camera
            camera_id: Camera device ID (e.g., 0 for /dev/video0)
            width: Desired frame width
            height: Desired frame height
            fps: Desired frames per second
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        self.cameras[name] = cap
        self.camera_settings[name] = {
            'id': camera_id,
            'width': width,
            'height': height,
            'fps': fps
        }
        
    def start(self):
        """Start all cameras."""
        for name, cap in self.cameras.items():
            if not cap.isOpened():
                settings = self.camera_settings[name]
                cap.open(settings['id'])
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to start camera {name}")
                    
                # Reapply settings
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['width'])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['height'])
                cap.set(cv2.CAP_PROP_FPS, settings['fps'])
                
    def stop(self):
        """Stop and release all cameras."""
        for cap in self.cameras.values():
            cap.release()
            
    def get_frame(self, camera_name: str) -> Tuple[Optional[np.ndarray], bool]:
        """Get a frame from the specified camera.
        
        Args:
            camera_name: Name of the camera to get frame from
            
        Returns:
            Tuple of (frame, success)
            - frame: numpy array of the frame if successful, None otherwise
            - success: True if frame was successfully captured
        """
        if camera_name not in self.cameras:
            return None, False
            
        cap = self.cameras[camera_name]
        success, frame = cap.read()
        return frame, success
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np
from threading import Thread, Lock
from queue import Queue
import time

class Camera:
    def __init__(self, camera_id: int, name: str, width: int = 640, height: int = 480, fps: int = 30):
        self.camera_id = camera_id
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=2)  # Keep only latest frame
        self.last_frame_time = 0
        self.lock = Lock()
        self._thread = None

    def start(self):
        """Start the camera capture thread."""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.name} (ID: {self.camera_id})")

        self.is_running = True
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        """Continuous capture loop running in a separate thread."""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    # Clear queue and put new frame
                    while not self.frame_queue.empty():
                        self.frame_queue.get()
                    self.frame_queue.put((frame, time.time()))
            time.sleep(1/self.fps)  # Control capture rate

    def get_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """Get the latest frame and its timestamp."""
        if self.frame_queue.empty():
            return None, 0
        return self.frame_queue.get()

    def stop(self):
        """Stop the camera capture."""
        self.is_running = False
        if self._thread:
            self._thread.join()
        if self.cap:
            self.cap.release()

class CameraManager:
    def __init__(self):
        self.cameras: Dict[str, Camera] = {}
        self.is_running = False

    def add_camera(self, name: str, camera_id: int, width: int = 640, height: int = 480, fps: int = 30):
        """Add a new camera to the manager."""
        if name in self.cameras:
            raise ValueError(f"Camera with name {name} already exists")
        
        camera = Camera(camera_id, name, width, height, fps)
        self.cameras[name] = camera
        if self.is_running:
            camera.start()

    def start(self):
        """Start all cameras."""
        self.is_running = True
        for camera in self.cameras.values():
            camera.start()

    def stop(self):
        """Stop all cameras."""
        self.is_running = False
        for camera in self.cameras.values():
            camera.stop()

    def get_frames(self) -> Dict[str, Tuple[Optional[np.ndarray], float]]:
        """Get the latest frames from all cameras."""
        return {name: camera.get_frame() for name, camera in self.cameras.items()}

    def get_frame(self, camera_name: str) -> Tuple[Optional[np.ndarray], float]:
        """Get the latest frame from a specific camera."""
        if camera_name not in self.cameras:
            raise ValueError(f"Camera {camera_name} not found")
        return self.cameras[camera_name].get_frame()
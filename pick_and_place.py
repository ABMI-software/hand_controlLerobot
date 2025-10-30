from typing import Optional, Tuple, Dict, List
import cv2
import numpy as np
from pathlib import Path
import time
import threading
from queue import Queue

from hand_teleop.cameras.camera_manager import CameraManager
from hand_teleop.detection.object_detector import ObjectDetector, DetectedObject
from hand_teleop.tracking.tracker import HandTracker
from hand_teleop.gripper_pose.gripper_pose import GripperPose

class PickingSystem:
    def __init__(
        self,
        arducam_id: int = 0,
        depth_camera_id: int = -1,  # -1 for OpenNI interface
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.5,
    ):
        # Initialize camera manager
        self.camera_manager = CameraManager()
        self.camera_manager.add_camera("arducam", arducam_id, width=1280, height=720)
        self.camera_manager.add_camera("depth", depth_camera_id, width=640, height=480)
        
        # Initialize object detector
        self.detector = ObjectDetector(model_path, conf_threshold)
        
        # Initialize hand tracker
        self.hand_tracker = HandTracker(cam_idx=depth_camera_id, model="wilor")
        
        # Control flags
        self.running = False
        self.mode = "detection"  # "detection", "auto_pick", "teleop"
        self.detected_objects: List[DetectedObject] = []
        self.selected_object: Optional[DetectedObject] = None
        
        # User interface
        self.display_queue = Queue()
        self._display_thread = None

    def start(self):
        """Start the picking system."""
        self.running = True
        self.camera_manager.start()
        
        # Start display thread
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()
        
        # Main control loop
        while self.running:
            if self.mode == "detection":
                self._run_detection()
            elif self.mode == "auto_pick":
                self._run_auto_pick()
            elif self.mode == "teleop":
                self._run_teleoperation()

            # Check for mode change commands
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
            elif key == ord('a'):
                self.mode = "auto_pick"
            elif key == ord('t'):
                self.mode = "teleop"

    def _run_detection(self):
        """Run object detection on Arducam feed."""
        frame, _ = self.camera_manager.get_frame("arducam")
        if frame is not None:
            # Detect objects
            self.detected_objects = self.detector.detect(frame)
            
            # Visualize detections
            viz_frame = self.detector.visualize(frame, self.detected_objects)
            
            # Add instructions
            cv2.putText(viz_frame, "Press 'a' for auto-pick or 't' for teleoperation",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Display frame
            self.display_queue.put(("Object Detection", viz_frame))

    def _run_auto_pick(self):
        """Run automatic picking mode."""
        if not self.detected_objects:
            print("No objects detected to pick")
            self.mode = "detection"
            return

        # Select nearest object if none selected
        if self.selected_object is None:
            self.selected_object = self.detected_objects[0]

        # TODO: Implement automatic picking sequence
        # This would include:
        # 1. Path planning to object
        # 2. Gripper positioning
        # 3. Grasping
        # 4. Object lifting
        print("Auto-pick mode not fully implemented")
        self.mode = "detection"

    def _run_teleoperation(self):
        """Run teleoperation mode."""
        # Get frames from both cameras
        arducam_frame, _ = self.camera_manager.get_frame("arducam")
        webcam_frame, _ = self.camera_manager.get_frame("webcam")

        if arducam_frame is not None and webcam_frame is not None:
            # Run hand tracking
            hand_pose = self.hand_tracker.track(webcam_frame)
            
            # Visualize both views
            if self.detected_objects:
                arducam_frame = self.detector.visualize(arducam_frame, self.detected_objects)
            
            # Combine frames for display
            combined_frame = np.hstack((arducam_frame, webcam_frame))
            self.display_queue.put(("Teleoperation", combined_frame))

    def _display_loop(self):
        """Display thread for showing camera feeds and UI."""
        while self.running:
            if not self.display_queue.empty():
                title, frame = self.display_queue.get()
                cv2.imshow(title, frame)
                cv2.waitKey(1)

    def stop(self):
        """Stop the picking system."""
        self.running = False
        if self._display_thread:
            self._display_thread.join()
        self.camera_manager.stop()
        cv2.destroyAllWindows()

def main():
    # Initialize picking system
    system = PickingSystem(
        arducam_id=0,  # Adjust as needed
        webcam_id=1,   # Adjust as needed
        model_path="yolov8n.pt",  # Use appropriate model
        conf_threshold=0.5
    )

    try:
        system.start()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        system.stop()

if __name__ == "__main__":
    main()
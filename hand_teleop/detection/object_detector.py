from typing import List, Optional, Tuple, Dict
import numpy as np
import cv2
from ultralytics import YOLO
from dataclasses import dataclass

@dataclass
class DetectedObject:
    """Class to store detection results."""
    id: int
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center: Tuple[float, float]  # center_x, center_y

class ObjectDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.5):
        """
        Initialize the YOLO object detector.
        
        Args:
            model_path: Path to the YOLO model weights
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects in the frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of DetectedObject instances
        """
        results = self.model(frame)[0]
        detections = []

        for i, detection in enumerate(results.boxes.data):
            x1, y1, x2, y2, conf, class_id = detection
            if conf < self.conf_threshold:
                continue

            # Get the center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Create DetectedObject instance
            detected_obj = DetectedObject(
                id=i,
                label=results.names[int(class_id)],
                confidence=float(conf),
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                center=(float(center_x), float(center_y))
            )
            detections.append(detected_obj)

        return detections

    def visualize(self, frame: np.ndarray, detections: List[DetectedObject]) -> np.ndarray:
        """
        Draw detection results on the frame.
        
        Args:
            frame: Input image frame
            detections: List of DetectedObject instances
            
        Returns:
            Frame with drawn detections
        """
        viz_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            
            # Draw bounding box
            cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label and confidence
            label = f"{det.label} {det.confidence:.2f}"
            cv2.putText(viz_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw center point
            center_x, center_y = map(int, det.center)
            cv2.circle(viz_frame, (center_x, center_y), 4, (0, 0, 255), -1)

        return viz_frame
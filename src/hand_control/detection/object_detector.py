"""
Object detection using YOLOv8.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Union
from ultralytics import YOLO

class ObjectDetector:
    """YOLOv8-based object detector with COCO classes."""
    
    # COCO class labels
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def __init__(self, model_path: Union[str, Path], conf_threshold: float = 0.5):
        """Initialize the object detector.
        
        Args:
            model_path: Path to YOLOv8 model file
            conf_threshold: Confidence threshold for detections (0-1)
        """
        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold
        
    def get_class_name(self, class_id: int) -> str:
        """Get the class name from COCO class ID.
        
        Args:
            class_id: Integer class index
            
        Returns:
            String class name
        """
        if 0 <= class_id < len(self.COCO_CLASSES):
            return self.COCO_CLASSES[class_id]
        return f"unknown_{class_id}"
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Run object detection on a frame.
        
        Args:
            frame: Input image as numpy array (BGR format)
            
        Returns:
            List of detections, each a dict with:
                - class: Class ID
                - confidence: Detection confidence
                - bbox: Bounding box [x1, y1, x2, y2]
        """
        results = self.model(frame, conf=self.conf_threshold)[0]
        detections = []
        
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            detections.append({
                'class': int(cls),
                'confidence': conf,
                'bbox': [int(x1), int(y1), int(x2), int(y2)]
            })
            
        return detections
        
    def visualize(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detection results on the frame with class labels.
        
        Args:
            frame: Input image
            detections: List of detections from detect()
            
        Returns:
            Frame with visualized detections
        """
        viz_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls = det['class']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(viz_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with class name and confidence
            label = f"{self.get_class_name(cls)}: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(viz_frame,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(viz_frame, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
        return viz_frame
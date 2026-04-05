import os
import gdown

folder_id = "1ojfHWoPK0U7pbhLZ2CiMx06yloqkZrGL"

# Create models directory
if not os.path.exists("models"):
    os.makedirs("models")

# Download models only if not already present
if len(os.listdir("models")) == 0:
    gdown.download_folder(
        id=folder_id,
        output="models",
        quiet=False
    )
"""
Road Condition Detector using Custom YOLO Model
Detects road cracks and damage: longitudinal crack, transverse crack, alligator crack, other corruption, pothole
"""
import os
from ultralytics import YOLO
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def model_path(name):
    return os.path.join("models", name)

class RoadConditionDetector:
    def __init__(self, model_name="crack_best.pt", conf_threshold=0.1):
        """
        Initialize road condition detector
        
        Args:
            model_name: YOLO model file name
            conf_threshold: Confidence threshold for detections (lowered to 0.1 for maximum sensitivity)
        """
        self.model_path = os.path.join(MODEL_DIR, model_name)
        self.conf_threshold = conf_threshold
        
        # Road condition class names
        self.class_names = {
            0: "longitudinal crack",
            1: "transverse crack",
            2: "alligator crack",
            3: "other corruption",
            4: "pothole"
        }
        
        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = YOLO(self.model_path)
        print(f"✅ Road Condition Detector loaded: {model_name} (threshold: {conf_threshold})")
    
    def detect(self, frame):
        """
        Detect road conditions in frame
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            List of detections: [(x1, y1, x2, y2, conf, cls_id, class_name), ...]
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_name = self.class_names.get(cls_id, f"unknown_{cls_id}")
            detections.append((x1, y1, x2, y2, conf, cls_id, class_name))
        
        return detections
    
    def get_severity(self, cls_id):
        """
        Get severity level of road condition
        
        Returns:
            'critical', 'high', or 'medium'
        """
        if cls_id == 4:  # pothole
            return "critical"
        elif cls_id in [2, 3]:  # alligator crack, other corruption
            return "high"
        else:  # longitudinal/transverse crack
            return "medium"

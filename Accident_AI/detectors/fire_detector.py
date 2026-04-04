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
Fire and Smoke Detector
Detects fire and smoke in frames (Phase-3 feature)
"""
import os
from ultralytics import YOLO
import numpy as np
import cv2

def model_path(name):
    return os.path.join("models", name)


class FireDetector:
    def __init__(self, model_name="best.pt", conf_threshold=0.2):
        """
        Initialize fire detector using custom YOLO model
        
        Args:
            model_name: YOLO model file name
            conf_threshold: Confidence threshold for detections (lowered for better detection)
        """
        self.model_path = os.path.join(MODEL_DIR, model_name)
        self.conf_threshold = conf_threshold
        
        # Load custom fire detection model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Fire detection model not found: {self.model_path}")
        
        self.model = YOLO(self.model_path)
        print(f"✅ Fire Detector loaded: {model_name} (threshold: {conf_threshold})")
    
    def detect(self, frame):
        """
        Detect fire in frame using YOLO model
        
        Args:
            frame: Input frame
            
        Returns:
            (has_fire, has_smoke, fire_regions)
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        fire_regions = []
        has_fire = False
        has_smoke = False  # Can be extended if model detects smoke separately
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Add fire region
            fire_regions.append((x1, y1, x2, y2, conf))
            has_fire = True
        
        return has_fire, has_smoke, fire_regions

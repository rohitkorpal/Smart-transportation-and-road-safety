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
Helmet Detector using Custom YOLO Model
Detects whether a person is wearing a helmet or not
"""
import os
from ultralytics import YOLO
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def model_path(name):
    return os.path.join("models", name)
    
class HelmetDetector:
    def __init__(self, model_name="helmet_best.pt", conf_threshold=0.2):
        """
        Initialize helmet detector
        
        Args:
            model_name: YOLO model file name
            conf_threshold: Confidence threshold for detections (lowered for better detection)
        """
        self.model_path = os.path.join(MODEL_DIR, model_name)
        self.conf_threshold = conf_threshold
        
        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = YOLO(self.model_path)
        print(f"✅ Helmet Detector loaded: {model_name} (threshold: {conf_threshold})")
    
    def detect(self, frame):
        """
        Detect helmets and people without helmets in frame
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Tuple of (with_helmet, without_helmet)
            - with_helmet: List of detections [(x1, y1, x2, y2, conf), ...]
            - without_helmet: List of detections [(x1, y1, x2, y2, conf), ...]
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        with_helmet = []
        without_helmet = []
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Assuming class 0 = with helmet, class 1 = without helmet
            # Adjust based on your model's actual class mapping
            if cls_id == 0:  # with helmet
                with_helmet.append((x1, y1, x2, y2, conf))
            else:  # without helmet
                without_helmet.append((x1, y1, x2, y2, conf))
        
        return with_helmet, without_helmet
    
    def count_violations(self, without_helmet):
        """
        Count number of helmet violations
        
        Args:
            without_helmet: List of detections without helmet
            
        Returns:
            Number of violations
        """
        return len(without_helmet)

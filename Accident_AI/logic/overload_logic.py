"""
Overload Vehicle Detection Logic
Detects vehicles carrying excessive load using YOLO object detection
"""
import numpy as np
import cv2

class OverloadDetector:
    def __init__(self, height_ratio_threshold=1.5, object_count_threshold=5):
        """
        Initialize overload detector
        
        Args:
            height_ratio_threshold: Ratio of detected objects height to vehicle height (default: 1.5)
            object_count_threshold: Minimum number of objects on vehicle to consider overload (default: 5)
        """
        self.height_ratio_threshold = height_ratio_threshold
        self.object_count_threshold = object_count_threshold
        self.overload_history = {}  # Track ID -> overload state
        print("Overload Detector initialized")
    
    def detect_overload(self, tracked_vehicles, yolo_detections, frame_shape):
        """
        Detect overloaded vehicles by analyzing objects on top of vehicles
        
        Args:
            tracked_vehicles: List of (x1, y1, x2, y2, track_id, conf, cls_id)
            yolo_detections: All YOLO detections from frame (including non-vehicle objects)
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            List of overloaded vehicle track IDs with metadata: [(track_id, reason, severity), ...]
        """
        overloaded_vehicles = []
        
        for vehicle in tracked_vehicles:
            x1, y1, x2, y2, track_id, conf, cls_id = vehicle
            vehicle_height = y2 - y1
            vehicle_width = x2 - x1
            vehicle_area = vehicle_height * vehicle_width
            
            # Skip very small detections
            if vehicle_area < 1000:
                continue
            
            # Count objects on or near the vehicle
            objects_on_vehicle = []
            
            for detection in yolo_detections:
                det_x1, det_y1, det_x2, det_y2, det_conf, det_cls = detection
                det_center_x = (det_x1 + det_x2) / 2
                det_center_y = (det_y1 + det_y2) / 2
                
                # Check if object is within vehicle's horizontal bounds
                if x1 <= det_center_x <= x2:
                    # Check if object is on top of vehicle (y coordinate less than vehicle top)
                    if det_y1 < y1 and det_y2 <= y2:
                        objects_on_vehicle.append(detection)
            
            # Analyze overload conditions
            overload_detected = False
            reason = ""
            severity = "medium"
            
            # Condition 1: Too many objects on vehicle
            if len(objects_on_vehicle) >= self.object_count_threshold:
                overload_detected = True
                reason = f"Multiple objects ({len(objects_on_vehicle)}) on vehicle"
                severity = "high"
            
            # Condition 2: Objects extend too high above vehicle
            if objects_on_vehicle:
                max_object_height = min([obj[1] for obj in objects_on_vehicle])  # Topmost y coordinate
                height_extension = y1 - max_object_height
                
                if height_extension > vehicle_height * (self.height_ratio_threshold - 1):
                    overload_detected = True
                    reason = f"Load extends {height_extension:.0f}px above vehicle"
                    severity = "critical"
            
            # Condition 3: Abnormal vehicle aspect ratio (very tall vehicle suggests overload)
            aspect_ratio = vehicle_height / vehicle_width if vehicle_width > 0 else 0
            if aspect_ratio > 2.0:  # Unusually tall vehicle
                overload_detected = True
                if not reason:
                    reason = f"Abnormal height ratio ({aspect_ratio:.2f})"
                severity = "high"
            
            if overload_detected:
                # Track overload history to avoid duplicate alerts
                if track_id not in self.overload_history:
                    self.overload_history[track_id] = True
                    overloaded_vehicles.append((track_id, reason, severity))
        
        return overloaded_vehicles
    
    def detect_overload_simple(self, tracked_vehicles, frame):
        """
        Simple overload detection based on vehicle dimensions and visual analysis
        
        Args:
            tracked_vehicles: List of (x1, y1, x2, y2, track_id, conf, cls_id)
            frame: Current frame for visual analysis
            
        Returns:
            List of overloaded vehicle track IDs: [(track_id, reason, severity), ...]
        """
        overloaded_vehicles = []
        
        for vehicle in tracked_vehicles:
            x1, y1, x2, y2, track_id, conf, cls_id = vehicle
            vehicle_height = y2 - y1
            vehicle_width = x2 - x1
            
            # Skip very small detections (lowered threshold for better detection)
            if vehicle_height < 30 or vehicle_width < 30:
                continue
            
            overload_detected = False
            reason = ""
            severity = "medium"
            
            # Method 1: Aspect ratio analysis
            aspect_ratio = vehicle_height / vehicle_width if vehicle_width > 0 else 0
            
            # Different thresholds for different vehicle types (LOWERED for better detection)
            if cls_id == 7:  # Truck
                if aspect_ratio > 1.3:  # Lowered from 1.8
                    overload_detected = True
                    reason = f"Truck height abnormal (ratio: {aspect_ratio:.2f})"
                    severity = "high"
            elif cls_id == 2:  # Car
                if aspect_ratio > 1.2:  # Lowered from 1.5
                    overload_detected = True
                    reason = f"Car overloaded (ratio: {aspect_ratio:.2f})"
                    severity = "critical"
            elif cls_id == 3:  # Motorcycle
                if aspect_ratio > 1.4:  # Lowered from 2.0
                    overload_detected = True
                    reason = f"Motorcycle overloaded (ratio: {aspect_ratio:.2f})"
                    severity = "critical"
            elif cls_id == 5:  # Bus - also check for overload
                if aspect_ratio > 1.6:
                    overload_detected = True
                    reason = f"Bus overloaded (ratio: {aspect_ratio:.2f})"
                    severity = "high"
            
            # Method 2: Visual density analysis (top portion of vehicle)
            if not overload_detected and y1 >= 0 and y2 < frame.shape[0]:
                # Analyze top 30% of vehicle
                top_region_height = int(vehicle_height * 0.3)
                if top_region_height > 10 and y1 - top_region_height >= 0:
                    top_region = frame[max(0, y1 - top_region_height):y1, x1:x2]
                    
                    if top_region.size > 0:
                        # Check for significant content above vehicle
                        gray_top = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray_top, 50, 150)
                        edge_density = np.sum(edges > 0) / edges.size
                        
                        if edge_density > 0.08:  # Lowered from 0.15 for better detection
                            overload_detected = True
                            reason = f"Objects detected above vehicle (density: {edge_density:.2f})"
                            severity = "medium"
            
            if overload_detected:
                # Track overload history
                if track_id not in self.overload_history:
                    self.overload_history[track_id] = True
                    overloaded_vehicles.append((track_id, reason, severity))
        
        return overloaded_vehicles
    
    def reset_history(self):
        """Reset overload detection history"""
        self.overload_history.clear()

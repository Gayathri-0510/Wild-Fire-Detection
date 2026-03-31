import cv2
import os
from ultralytics import YOLO
from src.utils.config import CONFIG
from src.model.predict import WildfireClassifier
from src.utils.helpers import draw_bounding_box

class ImageDetector:
    def __init__(self):
        self.classifier = WildfireClassifier()
        self.yolo_model_path = CONFIG['model']['detection_model']
        self.yolo_model = None
        
    def load_yolo(self):
        if self.yolo_model is None:
            if not os.path.exists(self.yolo_model_path):
                print(f"Warning: YOLOv8 model not found at {self.yolo_model_path}")
                return False
            # Suppress logs for YOLO in continuous inference
            self.yolo_model = YOLO(self.yolo_model_path)
        return True
            
    def detect(self, image_path, output_path=None):
        """Pipeline: MobileNetV2 classification -> YOLOv8 localization"""
        # Step 1: Classify with MobileNetV2
        has_fire, confidence = self.classifier.predict(image_path)
        
        # Read image to process and return
        image = cv2.imread(image_path)
        if image is None:
            return None, False, []
            
        detections = []
        
        # Step 2: YOLOv8 Regional Detection (only if MobileNetV2 detects fire)
        if has_fire:
            if self.load_yolo():
                results = self.yolo_model(image_path, verbose=False)[0]
                for box in results.boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    # Assuming class 0 is Fire
                    detections.append({'bbox': bbox, 'confidence': conf})
                    image = draw_bounding_box(image, bbox, f"Fire {conf:.2f}")
            else:
                # Fallback: OpenCV Color-Contour Detection since YOLO model is absent
                cv2.putText(image, "FIRE DETECTED (Color Fallback BBox)", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                import numpy as np
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                # Define range for red, orange, yellow in HSV
                lower_bound = np.array([0, 120, 150]) 
                upper_bound = np.array([35, 255, 255])
                
                mask = cv2.inRange(hsv, lower_bound, upper_bound)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 100: # filter noise
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        bbox = [x, y, x+w, y+h]
                        conf = confidence if 'confidence' in locals() else 0.95
                        detections.append({'bbox': bbox, 'confidence': conf})
                        
                        # Use the existing drawing function (draws RED by default)
                        image = draw_bounding_box(image, bbox, f"Fire {conf:.2f}")
        
        if output_path is not None:
             cv2.imwrite(output_path, image)
             
        return image, has_fire, detections

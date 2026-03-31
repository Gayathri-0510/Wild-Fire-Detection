import cv2
import os
from src.detection.image_detection import ImageDetector
from src.utils.config import CONFIG

class VideoDetector:
    def __init__(self, alert_callback=None):
        self.detector = ImageDetector()
        self.consecutive_frames_alert = CONFIG['model']['consecutive_frames_alert']
        self.fire_frames_count = 0
        self.alert_callback = alert_callback
        self.alert_sent = False
        
    def process_frame(self, frame):
        # We need to save the frame temporarily so MobileNetV2 and YOLOv8 can read it from path
        # Alternatively we can rewrite predict methods to accept numpy arrays
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)
        
        # Run detection pipeline
        processed_frame, has_fire, detections = self.detector.detect(temp_path)
        
        if has_fire:
            self.fire_frames_count += 1
            # Check for alert criteria
            if self.fire_frames_count >= self.consecutive_frames_alert and not self.alert_sent:
                print(">>> FIRE CONSECUTIVELY DETECTED. TRIGGERING ALERT SYSTEM <<<")
                if self.alert_callback:
                    self.alert_callback()
                self.alert_sent = True
        else:
            self.fire_frames_count = 0 
            
        # Draw status text
        status_color = (0, 0, 255) if has_fire else (0, 255, 0)
        status_text = "Status: FIRE" if has_fire else "Status: NORMAL"
        cv2.putText(processed_frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        if self.fire_frames_count > 0:
             cv2.putText(processed_frame, f"Consecutive Fire Frames: {self.fire_frames_count}/{self.consecutive_frames_alert}", 
                         (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                         
        # Cleanup
        if os.path.exists(temp_path):
             os.remove(temp_path)
             
        return processed_frame
        
    def process_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(video_path)
        
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = self.process_frame(frame)
            
            if out:
                out.write(processed_frame)
                
            cv2.imshow('Wildfire Video Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

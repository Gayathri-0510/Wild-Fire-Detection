import cv2
import sys
import os

# Add root directory to python path for module running
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detection.video_detection import VideoDetector
from src.alerts.sms_alert import send_sms_alert
from src.alerts.email_alert import send_email_alert

def trigger_alerts():
    print("Executing emergency alerts...")
    msg = "EMERGENCY: Wildfire detected from CCTV feed for 10 consecutive frames!"
    send_sms_alert(msg)
    send_email_alert("Urgent: Wildfire Alert", msg)

def start_live_monitoring(camera_index=0):
    print(f"Starting live CCTV monitoring from camera {camera_index}...")
    
    # Initialize detector with the alert trigger callback
    detector = VideoDetector(alert_callback=trigger_alerts)
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}. Ensure your webcam is connected.")
        return
        
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break
            
        # Process the frame
        processed_frame = detector.process_frame(frame)
        
        # Display the resulting frame
        cv2.imshow('Live CCTV Wildfire Monitoring', processed_frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_live_monitoring()

import argparse
from app.app import app
from src.utils.config import CONFIG

def main():
    parser = argparse.ArgumentParser(description="Wildfire Detection System Entry Point")
    parser.add_argument('--mode', choices=['web', 'camera', 'train', 'evaluate'], default='web',
                        help="Run mode: 'web' for Flask App, 'camera' for Live CCTV, 'train' to retrain model, 'evaluate' to show test accuracy")
    
    args = parser.parse_args()

    if args.mode == 'web':
        print("[*] Starting Premium Web Interface on http://localhost:5000 ...")
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    elif args.mode == 'camera':
        print("[*] Launching Realtime Camera Detection...")
        from realtime.live_detection import start_live_monitoring
        start_live_monitoring(camera_index=0)
        
    elif args.mode == 'train':
        print("[*] Launching Transfer Learning Process...")
        from src.model.train import train_model
        train_model()
        
    elif args.mode == 'evaluate':
        print("[*] Evaluating Model on Validation Dataset...")
        from src.model.evaluate import evaluate_model
        evaluate_model()

if __name__ == "__main__":
    main()

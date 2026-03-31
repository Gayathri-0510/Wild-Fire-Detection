import argparse
from app.app import app
from src.utils.config import CONFIG


def main():
    """
    Main entry point for Wildfire Detection System.
    Supports multiple modes:
    - web       : Launch Flask web interface
    - camera    : Start live CCTV / webcam detection
    - train     : Retrain MobileNetV2 model
    - evaluate  : Evaluate model performance
    """

    parser = argparse.ArgumentParser(
        description="Wildfire Detection System Entry Point"
    )

    parser.add_argument(
        '--mode',
        choices=['web', 'camera', 'train', 'evaluate'],
        default='web',
        help=(
            "Choose run mode:\n"
            "'web'      -> Start Flask Web App\n"
            "'camera'   -> Start Live Camera Monitoring\n"
            "'train'    -> Retrain Model\n"
            "'evaluate' -> Evaluate Model Accuracy"
        )
    )

    args = parser.parse_args()

    if args.mode == 'web':
        print("[*] Starting Premium Web Interface...")
        print("[*] Open in browser: http://localhost:5000")
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True
        )

    elif args.mode == 'camera':
        print("[*] Launching Realtime Camera Detection...")
        from realtime.live_detection import start_live_monitoring
        start_live_monitoring(camera_index=0)

    elif args.mode == 'train':
        print("[*] Starting Transfer Learning Model Training...")
        from src.model.train import train_model
        train_model()

    elif args.mode == 'evaluate':
        print("[*] Evaluating Model on Validation Dataset...")
        from src.model.evaluate import evaluate_model
        evaluate_model()

    else:
        print("[!] Invalid mode selected.")


if __name__ == "__main__":
    main()
from ultralytics import YOLO
import cv2
import os

# This auto-downloads the pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")


def detect_fire_region(image_path):
    img = cv2.imread(image_path)

    results = model(image_path)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])

            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    output_path = image_path.replace(".", "_boxed.")
    cv2.imwrite(output_path, img)

    return output_path
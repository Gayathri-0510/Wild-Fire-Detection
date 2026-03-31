import os
import cv2
import numpy as np
import tensorflow as tf

def preprocess_image_for_mobilenet(img_path, target_size=(224, 224)):
    """Preprocess image for MobileNetV2 classification."""
    image = cv2.imread(img_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    return image_array

def draw_bounding_box(image, bbox, label, color=(0, 0, 255), thickness=2):
    """Draw bounding box on image from YOLO output."""
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return image

def ensure_dir(dir_path):
    """Ensure that a directory exists."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

import sys
import os
import tensorflow as tf
import numpy as np

sys.path.append(os.path.abspath('c:/Users/Bhava/Desktop/Wildfire_Detection/Wild-Fire-Detection'))
from src.utils.helpers import preprocess_image_for_mobilenet
from src.utils.config import CONFIG

model_path = os.path.join('c:/Users/Bhava/Desktop/Wildfire_Detection/Wild-Fire-Detection', CONFIG['model']['classification_model'])
print(f"Loading model: {model_path}")

model = tf.keras.models.load_model(model_path)
print("Model created.")

img_path = 'c:/Users/Bhava/Desktop/Wildfire_Detection/Wild-Fire-Detection/app/static/uploads/OIP.jpg'
print(f"Predicting on {img_path}")
preprocessed_img = preprocess_image_for_mobilenet(img_path)

if preprocessed_img is not None:
    prediction = model.predict(preprocessed_img)[0]
    print(f"Prediction raw array: {prediction}")
else:
    print("Preprocessed img is None.")

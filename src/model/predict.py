import os
import tensorflow as tf
import numpy as np
from src.utils.config import CONFIG
from src.utils.helpers import preprocess_image_for_mobilenet

class WildfireClassifier:
    def __init__(self):
        self.model_path = CONFIG['model']['classification_model']
        self.model = None
        
    def load_model(self):
        if self.model is None:
            if not os.path.exists(self.model_path):
                 print(f"Warning: Classification model not found at {self.model_path}")
                 return False
            self.model = tf.keras.models.load_model(self.model_path)
        return True
            
    def predict(self, image_path):
        """Predicts if an image contains fire or not."""
        if not self.load_model():
             return False, 0.0
             
        preprocessed_img = preprocess_image_for_mobilenet(image_path)
        
        if preprocessed_img is None:
            return False, 0.0
            
        prediction = self.model.predict(preprocessed_img)[0]
        # Depending on class sorting: e.g. 'fire' is index 0 vs 'non_fire' index 1
        # For simplicity, returning the raw prediction array if needed
        # Assuming index 0 is fire
        fire_prob = float(prediction[0]) 
        has_fire = fire_prob > CONFIG['model']['fire_confidence_threshold']
        
        return has_fire, fire_prob

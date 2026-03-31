import tensorflow as tf
from src.utils.config import CONFIG
from src.preprocessing.preprocess import get_data_generators
import os

def evaluate_model():
    model_path = CONFIG['model']['classification_model']
    _, val_dataset = get_data_generators()
    
    if val_dataset is None:
        print("Validation dataset not available.")
        return
        
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist.")
        return
        
    try:
        model = tf.keras.models.load_model(model_path)
        loss, accuracy = model.evaluate(val_dataset)
        print(f"Validation Loss: {loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
    except Exception as e:
        print(f"Error evaluating model: {e}")

if __name__ == "__main__":
    evaluate_model()

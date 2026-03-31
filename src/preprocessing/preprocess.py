import os
import tensorflow as tf
from src.utils.config import CONFIG

def get_data_generators():
    """Create data generators for training and validation."""
    data_dir = CONFIG['paths']['raw_data']
    img_size = tuple(CONFIG['model']['img_size'])
    batch_size = CONFIG['model']['batch_size']
    
    # We assume 'data/raw' has subdirectories for classes like 'fire' and 'non_fire'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir if os.path.exists(train_dir) else data_dir,
        validation_split=0.2 if not os.path.exists(train_dir) else None,
        subset="training" if not os.path.exists(train_dir) else None,
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical' if os.path.exists(data_dir) else None
    )
    
    val_dataset = None
    if train_dataset is not None:
        val_dataset = tf.keras.utils.image_dataset_from_directory(
            val_dir if os.path.exists(val_dir) else data_dir,
            validation_split=0.2 if not os.path.exists(val_dir) else None,
            subset="validation" if not os.path.exists(val_dir) else None,
            seed=123,
            image_size=img_size,
            batch_size=batch_size,
            label_mode='categorical'
        )
        
        # Apply MobileNetV2 preprocessing
        preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
        class_names = train_dataset.class_names
        train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y))
        train_dataset.class_names = class_names
        val_dataset = val_dataset.map(lambda x, y: (preprocess_input(x), y))
    
    return train_dataset, val_dataset

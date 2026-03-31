import os
import tensorflow as tf
# Using fully qualified tf.keras patterns instead
from src.preprocessing.preprocess import get_data_generators
from src.utils.config import CONFIG
from src.utils.helpers import ensure_dir

def build_model(num_classes=2):
    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet', 
        include_top=False, 
        input_shape=(*CONFIG['model']['img_size'], 3)
    )
    # Freeze base model
    base_model.trainable = False
    
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    train_dataset, val_dataset = get_data_generators()
    
    if train_dataset is None:
        print("Error: Dataset not found in data/raw. Please download the Kaggle dataset into data/raw.")
        return

    # Check number of classes dynamically
    class_names = train_dataset.class_names
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    model = build_model(num_classes)
    
    models_dir = CONFIG['paths']['models_dir']
    ensure_dir(models_dir)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CONFIG['model']['classification_model'], save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
    
    print("Starting training...")
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=20, callbacks=callbacks)
    print(f"Training complete. Model saved to {CONFIG['model']['classification_model']}")

if __name__ == '__main__':
    train_model()

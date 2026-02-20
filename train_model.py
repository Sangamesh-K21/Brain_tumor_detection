"""
Brain Tumor Detection - Model Training Script
Trains a CNN model using your own dataset
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ====================================
# CONFIGURATION
# ====================================
DATASET_PATH = "dataset"  # Change this to your dataset folder path
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
TEST_SPLIT = 0.2

# Class names (update based on your dataset)
class_names = [
    "Glioma Tumor",
    "Meningioma Tumor", 
    "No Tumor",
    "Pituitary Tumor"
]

# ====================================
# DATASET STRUCTURE EXPECTED:
# dataset/
# ├── Glioma Tumor/
# │   ├── img1.jpg
# │   ├── img2.jpg
# │   └── ...
# ├── Meningioma Tumor/
# │   ├── img1.jpg
# │   └── ...
# ├── No Tumor/
# ├── Pituitary Tumor/
# ====================================

def build_model():
    """Build CNN model"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(len(class_names), activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    """Train the model"""
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset folder '{DATASET_PATH}' not found!")
        print("\n📁 Please create this folder structure:")
        print("dataset/")
        for class_name in class_names:
            print(f"  ├── {class_name}/")
            print("  │   ├── img1.jpg")
            print("  │   ├── img2.jpg")
            print("  │   └── ...")
        return
    
    # Image Data Generator with Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=TEST_SPLIT
    )
    
    # Load training data
    print("📂 Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    # Load validation data
    validation_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    # Build and train model
    print("🧠 Building model...")
    model = build_model()
    model.summary()
    
    print("\n⏳ Training model...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )
    
    # Save model
    model_path = "keras_model.h5"
    model.save(model_path)
    print(f"\n✅ Model saved to {model_path}")
    
    # Plot training history
    plot_history(history)
    
    return model, history

def plot_history(history):
    """Plot training and validation metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("📊 Training history saved to training_history.png")
    plt.show()

if __name__ == "__main__":
    print("🔬 Brain Tumor Detection - Model Training\n")
    train_model()

"""
Brain Tumor Detection - Model Testing/Prediction Script
Test your trained model on new images
"""

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import argparse

MODEL_PATH = "keras_model.h5"

class_names = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor"
]

def preprocess_image(img_path):
    """Preprocess image for model prediction"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    img_arr = img_arr / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

def predict_image(img_path):
    """Predict tumor class for a single image"""
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file '{MODEL_PATH}' not found!")
        print("Run 'python train_model.py' first to train the model.")
        return
    
    if not os.path.exists(img_path):
        print(f"❌ Image file '{img_path}' not found!")
        return
    
    # Load model
    print(f"🧠 Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    # Preprocess image
    print(f"🖼️  Processing image: {img_path}")
    img_arr = preprocess_image(img_path)
    
    # Make prediction
    print("⏳ Making prediction...")
    predictions = model.predict(img_arr)
    class_id = np.argmax(predictions[0])
    confidence = predictions[0][class_id] * 100
    
    # Display results
    print("\n" + "="*50)
    print("🔬 PREDICTION RESULTS")
    print("="*50)
    print(f"Predicted Class: {class_names[class_id]}")
    print(f"Confidence: {confidence:.2f}%")
    print("\nAll Probabilities:")
    for i, class_name in enumerate(class_names):
        prob = predictions[0][i] * 100
        print(f"  • {class_name}: {prob:.2f}%")
    print("="*50)

def batch_predict(folder_path):
    """Predict for all images in a folder"""
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file '{MODEL_PATH}' not found!")
        return
    
    if not os.path.isdir(folder_path):
        print(f"❌ Folder '{folder_path}' not found!")
        return
    
    # Load model
    print(f"🧠 Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir(folder_path):
        if os.path.splitext(file)[1].lower() in image_extensions:
            image_files.append(os.path.join(folder_path, file))
    
    if not image_files:
        print(f"❌ No images found in '{folder_path}'")
        return
    
    print(f"\n📊 Found {len(image_files)} images. Processing...")
    print("="*60)
    
    results = []
    
    for idx, img_path in enumerate(image_files, 1):
        try:
            img_arr = preprocess_image(img_path)
            predictions = model.predict(img_arr, verbose=0)
            class_id = np.argmax(predictions[0])
            confidence = predictions[0][class_id] * 100
            
            results.append({
                'image': os.path.basename(img_path),
                'prediction': class_names[class_id],
                'confidence': confidence
            })
            
            print(f"{idx}. {os.path.basename(img_path)}")
            print(f"   Prediction: {class_names[class_id]} ({confidence:.2f}%)")
            
        except Exception as e:
            print(f"{idx}. {os.path.basename(img_path)} - ERROR: {str(e)}")
    
    print("="*60)
    print(f"\n✅ Processed {len(results)} images successfully!")
    
    # Summary
    class_counts = {}
    for result in results:
        pred = result['prediction']
        class_counts[pred] = class_counts.get(pred, 0) + 1
    
    print("\n📈 Summary:")
    for class_name, count in class_counts.items():
        print(f"  • {class_name}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Brain Tumor Detection - Model Testing')
    parser.add_argument('--image', '-i', help='Path to single image for prediction')
    parser.add_argument('--batch', '-b', help='Path to folder with images for batch prediction')
    
    args = parser.parse_args()
    
    if args.image:
        predict_image(args.image)
    elif args.batch:
        batch_predict(args.batch)
    else:
        print("🧠 Brain Tumor Detection - Model Testing")
        print("\nUsage:")
        print("  Single image:  python test_model.py -i path/to/image.jpg")
        print("  Batch test:    python test_model.py -b path/to/folder")
        print("\nExamples:")
        print("  python test_model.py -i test_image.jpg")
        print("  python test_model.py -b dataset/Glioma Tumor")

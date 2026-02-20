# 🧠 Brain Tumor Detection - Dataset Guide

## Dataset Structure

Your dataset should be organized like this:

```
dataset/
├── Glioma Tumor/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   └── ... (more images)
├── Meningioma Tumor/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── No Tumor/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Pituitary Tumor/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## How to Prepare Your Dataset

### 1. **Collect Brain MRI Images**
   - Brain MRI scan images in grayscale or color
   - Supported formats: `.jpg`, `.png`, `.jpeg`, `.bmp`
   - Recommended: At least 50-100 images per class
   - Size: Any size (will be resized to 224x224)

### 2. **Organize into Folders**
   ```powershell
   # Create the main dataset folder
   mkdir dataset
   
   # Create class folders
   mkdir "dataset/Glioma Tumor"
   mkdir "dataset/Meningioma Tumor"
   mkdir "dataset/No Tumor"
   mkdir "dataset/Pituitary Tumor"
   
   # Copy images into respective folders
   ```

### 3. **Download Sample Datasets**

#### Option A: **Kaggle Brain Tumor MRI Dataset**
   - Download from: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
   - Steps:
     1. Create Kaggle account
     2. Download the dataset
     3. Extract and organize as shown above

#### Option B: **Brain Tumor MRI Dataset (Multimodal)**
   - Download from: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
   - Follow the folder structure above

#### Option C: **Use Your Own Medical Images**
   - Collect from hospitals (with permission)
   - Ensure patient privacy and HIPAA compliance
   - Get proper labeling from radiologists

### 4. **Train the Model**

```powershell
# Navigate to the app directory
cd C:\Users\User\Downloads\bran_app\bran_app

# Activate virtual environment
.\env\Scripts\Activate.ps1

# Run the training script
python train_model.py
```

**Output:**
- ✅ New `keras_model.h5` (trained model)
- 📊 `training_history.png` (accuracy/loss graphs)

### 5. **Update the App**
The app will automatically use the new `keras_model.h5` file!

```powershell
# Start the app
python app.py
```

## Important Notes

⚠️ **Data Requirements:**
- **Minimum images per class:** 50-100
- **Recommended:** 200-500 per class
- **Total:** 800-2000 images for good results
- **Quality:** Clear, properly labeled images

📌 **Tips for Better Results:**
1. **Data Balance:** Keep similar number of images per class
2. **Image Quality:** Use clear, high-resolution MRI scans
3. **Preprocessing:** Ensure images are properly labeled
4. **Augmentation:** Script includes random rotations, flips, and zooms
5. **Validation:** Use 80% training, 20% validation split

🔬 **Training Parameters (configurable in `train_model.py`):**
- **IMG_SIZE:** 224×224 pixels
- **BATCH_SIZE:** 32 images per batch
- **EPOCHS:** 20 training iterations
- **LEARNING_RATE:** 0.001 (Adam optimizer)

## Troubleshooting

### "Dataset folder not found"
```powershell
# Make sure dataset folder exists
if (!(Test-Path "dataset")) { mkdir dataset }
```

### "No images found in class folders"
```powershell
# Check folder contents
Get-ChildItem "dataset/Glioma Tumor" -File
```

### "Low accuracy after training"
- Add more images to your dataset
- Increase `EPOCHS` to 30-40 in `train_model.py`
- Check image quality and proper labeling

### "Out of memory error"
- Reduce `BATCH_SIZE` from 32 to 16
- Reduce images resolution in training script

## Model Architecture

The CNN model has:
- **4 Convolutional Blocks** with ReLU activation
- **MaxPooling Layers** for dimensionality reduction
- **Flatten Layer** to convert to 1D
- **Dense Layers** with Dropout (50%) for classification
- **Softmax Output** for probabilities

## Next Steps

1. ✅ Organize your dataset
2. ✅ Run `python train_model.py`
3. ✅ Check `training_history.png` for performance
4. ✅ Upload new model to GitHub
5. ✅ Test with the Flask web app

---

**Good luck training your model! 🚀**

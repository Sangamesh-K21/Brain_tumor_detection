# 🧠 Brain Tumor Detection Application

A deep learning-based Flask web application for detecting and classifying brain tumors from MRI images using Keras/TensorFlow.

## 📋 Features

✅ **User Authentication** - Register & Login system with JSON-based user storage  
✅ **MRI Image Upload** - Upload brain scan images through web interface  
✅ **Tumor Classification** - Detect 4 tumor types:
  - Glioma Tumor
  - Meningioma Tumor
  - No Tumor
  - Pituitary Tumor

✅ **Image Validation** - Automatic MRI image verification  
✅ **Confidence Scores** - Display probability for each classification  
✅ **Model Retraining** - Easy scripts to train with your own dataset  

## 🚀 Quick Start

### 1. **Install Dependencies**

```powershell
# Navigate to project directory
cd c:\Users\User\Downloads\bran_app\bran_app

# Activate virtual environment
.\env\Scripts\Activate.ps1

# Install requirements (if not already installed)
pip install -r requirements.txt
```

### 2. **Run the Application**

```powershell
python app.py
```

Then open your browser and go to: **http://127.0.0.1:5000**

### 3. **Use the App**

1. **Landing Page** - Overview of the app
2. **Register** - Create new account
3. **Login** - Sign in with credentials
4. **Upload MRI** - Upload brain scan image
5. **Get Results** - Receive tumor classification with confidence

## 📊 Files & Structure

```
bran_app/
├── app.py                  # Main Flask application
├── train_model.py          # Model training script
├── test_model.py           # Model testing/prediction script
├── keras_model.h5          # Pre-trained Keras model
├── requirements.txt        # Python dependencies
├── labels.txt              # Class names
├── users.json              # User database
├── DATASET_GUIDE.md        # Dataset preparation guide
├── README.md               # This file
├── templates/              # HTML templates
│   ├── landing.html        # Landing page
│   ├── login.html          # Login page
│   ├── register.html       # Registration page
│   └── index.html          # Upload & prediction page
├── static/                 # Static files
│   └── uploads/            # Uploaded images storage
└── env/                    # Virtual environment
```

## 🤖 Training Your Own Model

### Step 1: Prepare Dataset

See [DATASET_GUIDE.md](DATASET_GUIDE.md) for detailed instructions on organizing your data.

Dataset structure:
```
dataset/
├── Glioma Tumor/
├── Meningioma Tumor/
├── No Tumor/
└── Pituitary Tumor/
```

### Step 2: Train the Model

```powershell
# Activate environment
.\env\Scripts\Activate.ps1

# Run training script
python train_model.py
```

**Output:**
- 📁 `keras_model.h5` - New trained model
- 📊 `training_history.png` - Training accuracy/loss graphs

### Step 3: Test the Model

```powershell
# Test single image
python test_model.py -i path/to/image.jpg

# Batch test multiple images
python test_model.py -b path/to/folder
```

## 🔧 Advanced Configuration

### Change Training Parameters

Edit `train_model.py`:
```python
DATASET_PATH = "dataset"      # Your dataset folder
IMG_SIZE = 224                # Image size (224x224)
BATCH_SIZE = 32               # Images per batch
EPOCHS = 20                   # Training iterations
TEST_SPLIT = 0.2              # 80% train, 20% validation
```

### Change Class Names

Update `class_names` in all Python files:
```python
class_names = [
    "Your Class 1",
    "Your Class 2",
    "Your Class 3",
    "Your Class 4"
]
```

## 📦 Dependencies

- **Flask** - Web framework
- **TensorFlow/Keras** - Deep learning
- **Pillow** - Image processing
- **NumPy** - Numerical computing
- **scikit-learn** - ML utilities

Install all with:
```powershell
pip install -r requirements.txt
```

## 🔄 GitHub Push

Push changes to GitHub:

```powershell
git add .
git commit -m "Update: description of changes"
git push
```

## 🐛 Troubleshooting

### App won't start
```powershell
# Check if environment is activated (should show (env) at prompt)
.\env\Scripts\Activate.ps1

# Check Python version
python --version

# Run with debug output
python app.py
```

### Model training fails
- Ensure `dataset/` folder exists with proper structure
- Check minimum 50+ images per class
- Increase `EPOCHS` if accuracy is low
- See [DATASET_GUIDE.md](DATASET_GUIDE.md) for troubleshooting

### Out of memory errors
- Reduce `BATCH_SIZE` from 32 to 16
- Reduce `IMG_SIZE` from 224 to 128
- Use fewer images for training

## 📚 Model Architecture

**CNN (Convolutional Neural Network):**
- 4 Convolutional blocks (32→64→128→128 filters)
- MaxPooling layers for feature reduction
- Dropout (50%) to prevent overfitting
- Dense layers for classification
- Softmax output for probability distribution

## 🎓 Learning Resources

- [TensorFlow/Keras Docs](https://www.tensorflow.org/api_docs)
- [CNN Image Classification](https://www.tensorflow.org/tutorials/images/cnn)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Medical Imaging Datasets](https://www.kaggle.com/datasets?tags=medical)

## 📝 Dataset Sources

- **Kaggle Brain Tumor Dataset**: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- **Brain Tumor Classification**: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

## ⚠️ Disclaimer

**This application is for educational purposes only.** It should not be used for actual medical diagnosis without proper validation by medical professionals. Always consult with qualified radiologists for tumor detection and diagnosis.

## 📄 License

This project is provided as-is for educational purposes.

## 👨‍💻 Author

**Sangamesh K**  
GitHub: [Sangamesh-K21](https://github.com/Sangamesh-K21)

---

**Happy training! 🚀**

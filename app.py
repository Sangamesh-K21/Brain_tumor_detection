from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import json

app = Flask(__name__)
app.secret_key = "super_secret_key"

# -------------------------------
# LOAD MODEL (UNCHANGED)
# -------------------------------
MODEL_PATH = "keras_model.h5"
model = load_model(MODEL_PATH)

class_names = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "No Tumor",
    "Pituitary Tumor"
]

# -------------------------------
# USER AUTH (JSON BASED)
# -------------------------------
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

# -------------------------------
# MRI IMAGE CHECK (UNCHANGED)
# -------------------------------
def is_mri_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((150, 150))
    arr = np.array(img)

    colored_pixels = 0
    total_pixels = arr.shape[0] * arr.shape[1]

    for pixel in arr.reshape(-1, 3):
        r, g, b = pixel
        if abs(r - g) > 25 or abs(r - b) > 25 or abs(g - b) > 25:
            colored_pixels += 1

    color_ratio = (colored_pixels / total_pixels) * 100
    return color_ratio < 10

# -------------------------------
# PREPROCESS IMAGE (UNCHANGED)
# -------------------------------
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.img_to_array(img)
    img_arr = img_arr / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

# -------------------------------
# LANDING PAGE
# -------------------------------
@app.route("/")
def landing():
    return render_template("landing.html")

# -------------------------------
# LOGIN
# -------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users = load_users()

        if username in users and users[username] == password:
            session["user"] = username
            return redirect(url_for("home"))

        return "Invalid username or password"

    return render_template("login.html")

# -------------------------------
# REGISTER
# -------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users = load_users()

        if username in users:
            return "User already exists"

        users[username] = password
        save_users(users)

        return redirect(url_for("login"))

    return render_template("register.html")

# -------------------------------
# LOGOUT
# -------------------------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# -------------------------------
# HOME (INDEX.HTML – UNCHANGED)
# -------------------------------
@app.route("/home")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")

# -------------------------------
# PREDICT ROUTE (UNCHANGED LOGIC)
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No image selected"

    upload_folder = "static/uploads"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # 1. MRI CHECK
    if not is_mri_image(file_path):
        return render_template(
            "index.html",
            prediction="Unknown Image – Please upload a proper Brain MRI image.",
            uploaded_image=file_path
        )

    # 2. MODEL PREDICTION
    img = preprocess_image(file_path)
    preds = model.predict(img)[0]
    class_id = np.argmax(preds)
    result = class_names[class_id]

    return render_template(
        "index.html",
        prediction=result,
        uploaded_image=file_path
    )

# -------------------------------
# RUN APP
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

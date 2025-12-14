# ============================================================
#                render.py (FINAL FOR RENDER)
# ============================================================

import os
import io
import json
import base64
import requests
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# ---------------- FIREBASE (ENV VAR ONLY) ----------------
import firebase_admin
from firebase_admin import credentials, firestore

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cnn_model4.h5")

# ---------------- FLASK APP ----------------
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ---------------- MODEL ----------------
model = None
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.6

CLASSES = [
    "Baybay Tall Coconut",
    "Catigan Dwarf Coconut",
    "Laguna Tall Coconut",
    "Tacunan Dwarf Coconut",
    "NotCoconut",
    "Unknown Dwarf Variety",
    "Unknown Tall Variety"
]

CLASS_INFO = {
    "Baybay Tall Coconut": {
        "class_name": "Baybay Tall Coconut",
        "lifespan": "60–90 years",
        "definition": "Tall coconut variety with strong trunk and high yield."
    },
    "Catigan Dwarf Coconut": {
        "class_name": "Catigan Dwarf Coconut",
        "lifespan": "60–90 years",
        "definition": "Dwarf variety known for early fruiting."
    },
    "Laguna Tall Coconut": {
        "class_name": "Laguna Tall Coconut",
        "lifespan": "60–90 years",
        "definition": "Tall variety adaptable to different environments."
    },
    "Tacunan Dwarf Coconut": {
        "class_name": "Tacunan Dwarf Coconut",
        "lifespan": "60–90 years",
        "definition": "Compact dwarf coconut with quality nuts."
    }
}

# ---------------- FIREBASE INIT ----------------
db = None
try:
    firebase_json = os.environ.get("FIREBASE_CREDENTIALS")

    if firebase_json:
        cred_dict = json.loads(firebase_json)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firebase connected via ENV VAR")
    else:
        print("⚠️ FIREBASE_CREDENTIALS not set")

except Exception as e:
    print("⚠️ Firebase init failed:", e)

# ---------------- IMAGE HELPERS ----------------
def load_image_from_base64(data):
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    decoded = base64.b64decode(data)
    return Image.open(io.BytesIO(decoded)).convert("RGB")

def load_image_from_url(url):
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def load_image_from_file(file):
    return Image.open(io.BytesIO(file.read())).convert("RGB")

def preprocess(img):
    img = img.resize(IMG_SIZE)
    arr = keras_image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# ---------------- PREDICTION ----------------
def predict_image(img):
    global model

    if model is None:
        print("📦 Loading CNN model...")
        model = load_model(MODEL_PATH)
        print("✅ Model loaded")

    x = preprocess(img)
    preds = model.predict(x, verbose=0)

    idx = int(np.argmax(preds))
    confidence = float(preds[0][idx])
    label = CLASSES[idx]

    if confidence < CONFIDENCE_THRESHOLD or "Unknown" in label or label == "NotCoconut":
        return {
            "class_name": "Invalid Image",
            "lifespan": "None",
            "definition": "None",
            "confidence": 0.0,
            "is_valid": False
        }

    info = CLASS_INFO[label]

    return {
        "class_name": info["class_name"],
        "lifespan": info["lifespan"],
        "definition": info["definition"],
        "confidence": round(confidence, 4),
        "is_valid": True
    }

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    img = None

    if request.is_json:
        data = request.get_json()
        if "image_base64" in data:
            img = load_image_from_base64(data["image_base64"])
        elif "image_url" in data:
            img = load_image_from_url(data["image_url"])

    if img is None and "image" in request.files:
        img = load_image_from_file(request.files["image"])

    if img is None:
        return jsonify({"error": "No image provided"}), 400

    result = predict_image(img)

    if db and result["is_valid"]:
        try:
            db.collection("CoconutPredictions").add(result)
        except Exception as e:
            print("⚠️ Firestore write failed:", e)

    return jsonify(result)

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

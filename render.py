# ============================================================
#                       render.py (FULL FIX)
# ============================================================

import os
import io
import json
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cnn_model4.h5")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ---------------- FLASK APP ----------------
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
CORS(app)

# ---------------- MODEL ----------------
model = None
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.4  # lower to allow predictions even if slightly uncertain

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
def load_image_from_file(file):
    return Image.open(io.BytesIO(file.read())).convert("RGB")

def preprocess_image(img):
    img = img.resize(IMG_SIZE)
    arr = keras_image.img_to_array(img) / 255.0
    return arr[np.newaxis, ...]

def predict_image(img):
    global model
    if model is None:
        print("📦 Loading CNN model...")
        model = load_model(MODEL_PATH)
        print("✅ Model loaded")

    x = preprocess_image(img)
    preds = model.predict(x, verbose=0)
    idx = int(preds.argmax())
    confidence = float(preds[0][idx])
    label = CLASSES[idx]

    # Always return prediction, even if low confidence
    if confidence < CONFIDENCE_THRESHOLD:
        label = f"Low Confidence: {label}"

    # If label not in CLASS_INFO, return basic info
    info = CLASS_INFO.get(label, {
        "class_name": label,
        "lifespan": "Unknown",
        "definition": "No info available"
    })

    return {
        "class_name": info["class_name"],
        "lifespan": info["lifespan"],
        "definition": info["definition"],
        "confidence": round(confidence, 4),
        "is_valid": label not in ["NotCoconut", "Unknown Dwarf Variety", "Unknown Tall Variety"]
    }

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    img = None
    location = None

    # JSON input (optional)
    if request.is_json:
        data = request.get_json()
        location = data.get("location", "Unknown")
        class_name = data.get("class_name")
        if class_name:
            info = CLASS_INFO.get(class_name, {
                "class_name": class_name,
                "lifespan": "Unknown",
                "definition": "No info available"
            })
            is_valid = class_name in CLASS_INFO
            result = {
                "class_name": info["class_name"],
                "lifespan": info["lifespan"],
                "definition": info["definition"],
                "location": location,
                "confidence": 1.0 if is_valid else 0.0,
                "is_valid": is_valid
            }
            if db and is_valid:
                try:
                    db.collection("CoconutPredictions").add(result)
                except Exception as e:
                    print("⚠️ Firestore write failed:", e)
            return jsonify(result)

    # File upload
    if "image" in request.files:
        img = load_image_from_file(request.files["image"])
        location = request.form.get("location", "Unknown")

    if img is None:
        return jsonify({"error": "No image provided"}), 400

    result = predict_image(img)
    result["location"] = location

    if db and result.get("is_valid", False):
        try:
            db.collection("CoconutPredictions").add(result)
        except Exception as e:
            print("⚠️ Firestore write failed:", e)

    return jsonify(result)

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running on port {port}")
    app.run(host="0.0.0.0", port=port)

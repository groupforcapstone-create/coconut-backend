# ============================================================
# server.py - Coconut Seedling Detection API
# ============================================================

import os
import io
import base64
import requests
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import firebase_admin
from firebase_admin import credentials, firestore

# -------------------- PATHS --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "cnn_model4.h5")
SERVICE_ACCOUNT_PATH = os.path.join(BASE_DIR, "serviceAccountKey.json")
TEMPLATES_PATH = os.path.join(BASE_DIR, "templates")
STATIC_PATH = os.path.join(BASE_DIR, "static")

# -------------------- FLASK APP --------------------
app = Flask(
    __name__,
    template_folder=TEMPLATES_PATH,
    static_folder=STATIC_PATH
)
CORS(app)

# -------------------- LOAD CNN MODEL --------------------
print("📦 Loading CNN model...")
model = load_model(MODEL_PATH)
print("✅ CNN model loaded")

# -------------------- FIREBASE INIT --------------------
db = None
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("✅ Firebase connected")
except Exception as e:
    print("⚠️ Firebase disabled:", e)

# -------------------- CONSTANTS --------------------
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
    "Baybay Tall Coconut": {"class_name": "Baybay Tall Coconut", "lifespan": "60–90 years", "definition": "Tall coconut variety with strong trunk and high yield."},
    "Catigan Dwarf Coconut": {"class_name": "Catigan Dwarf Coconut", "lifespan": "60–90 years", "definition": "Dwarf variety known for early fruiting."},
    "Laguna Tall Coconut": {"class_name": "Laguna Tall Coconut", "lifespan": "60–90 years", "definition": "Tall variety adaptable to different environments."},
    "Tacunan Dwarf Coconut": {"class_name": "Tacunan Dwarf Coconut", "lifespan": "60–90 years", "definition": "Compact dwarf coconut with quality nuts."},
    "Unknown Tall": {"class_name": "Unknown Tall Coconut", "lifespan": "Unknown", "definition": "Possibly a tall coconut variety."},
    "Unknown Dwarf": {"class_name": "Unknown Dwarf Coconut", "lifespan": "Unknown", "definition": "Possibly a dwarf coconut variety."},
    "NotCoconut": {"class_name": "Invalid Image", "lifespan": "None", "definition": "Uploaded image is not a coconut seedling."}
}

# -------------------- IMAGE HELPERS --------------------
def pil_from_base64(data):
    """Convert base64 string to PIL Image"""
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    decoded = base64.b64decode(data)
    return Image.open(io.BytesIO(decoded)).convert("RGB")

def pil_from_url(url):
    """Load image from URL"""
    r = requests.get(url, timeout=8)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

def pil_from_file(file_storage):
    """Load image from uploaded file"""
    return Image.open(io.BytesIO(file_storage.read())).convert("RGB")

def preprocess(img):
    """Resize and normalize image for model"""
    img = img.resize(IMG_SIZE)
    arr = keras_image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# -------------------- PREDICTION --------------------
def predict_image(img):
    x = preprocess(img)
    preds = model.predict(x, verbose=0)
    
    idx = int(np.argmax(preds))
    conf = float(preds[0][idx])
    label = CLASSES[idx]

    # Handle unknown and low-confidence predictions
    if label in ["Unknown Dwarf Variety", "Unknown Tall Variety"]:
        label = "Unknown Dwarf" if "Dwarf" in label else "Unknown Tall"
        conf = 0.55

    if label == "NotCoconut" or conf < CONFIDENCE_THRESHOLD:
        label = "NotCoconut"
        conf = 0.0

    info = CLASS_INFO.get(label, {})
    return {
        "class_name": info.get("class_name", label),
        "lifespan": info.get("lifespan", "N/A"),
        "definition": info.get("definition", "N/A"),
        "confidence": round(conf, 4),
        "is_valid": label not in ["Unknown Dwarf", "Unknown Tall", "NotCoconut"]
    }

# -------------------- ROUTES --------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard.html")
def dashboard():
    return render_template("dashboard.html")

@app.route("/admin.html")
def admin():
    return render_template("admin.html")

@app.route("/register.html")
def register():
    return render_template("register.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_PATH, filename)

@app.route("/predict", methods=["POST"])
def predict():
    img = None
    source = None

    # JSON requests
    if request.is_json:
        data = request.get_json()
        if "image_base64" in data:
            img = pil_from_base64(data["image_base64"])
            source = "base64"
        elif "image_url" in data:
            img = pil_from_url(data["image_url"])
            source = "url"

    # Multipart form upload
    if img is None and "image" in request.files:
        img = pil_from_file(request.files["image"])
        source = "multipart"

    if img is None:
        return jsonify({"error": "No image provided"}), 400

    result = predict_image(img)
    result["image_source"] = source

    # Save to Firebase if enabled
    if db and result["is_valid"]:
        try:
            db.collection("CoconutPredictions").add(result)
        except Exception as e:
            print("⚠️ Firestore save failed:", e)

    return jsonify(result)

# -------------------- HEALTH CHECK --------------------
@app.route("/healthz", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

# -------------------- RUN --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running on port {port}")
    app.run(host="0.0.0.0", port=port)

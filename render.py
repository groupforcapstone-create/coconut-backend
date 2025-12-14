# ============================================================
#                       render.py (RENDER-FRIENDLY)
# ============================================================

import os
import io
import json
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# ---------------- FLASK SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_PATH = os.path.join(BASE_DIR, "templates")
STATIC_PATH = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "cnn_model4.h5")

app = Flask(__name__, template_folder=TEMPLATES_PATH, static_folder=STATIC_PATH)
CORS(app)

# ---------------- FIREBASE SETUP ----------------
db = None
try:
    import firebase_admin
    from firebase_admin import credentials, firestore

    firebase_json = os.environ.get("FIREBASE_CREDENTIALS")
    if firebase_json:
        cred_dict = json.loads(firebase_json)
        cred = credentials.Certificate(cred_dict)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firebase connected.")
    else:
        print("⚠️ FIREBASE_CREDENTIALS not set. Firebase disabled.")
except Exception as e:
    print(f"⚠️ Firebase setup failed: {e}")
    db = None

# ---------------- CONSTANTS ----------------
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
    "Baybay Tall Coconut": {"class_name": "Baybay Tall Coconut", "lifespan": "60-90 years",
                            "definition": "A tall coconut variety commonly grown for its strong trunk and high yield."},
    "Catigan Dwarf Coconut": {"class_name": "Catigan Dwarf Coconut", "lifespan": "60-90 years",
                              "definition": "A dwarf coconut variety known for early fruiting and consistent nut production."},
    "Laguna Tall Coconut": {"class_name": "Laguna Tall Coconut", "lifespan": "60-90 years",
                            "definition": "A tall coconut variety recognized for its durability and adaptability."},
    "Tacunan Dwarf Coconut": {"class_name": "Tacunan Dwarf Coconut", "lifespan": "60-90 years",
                              "definition": "A compact dwarf coconut variety valued for its high-quality nuts."},
    "Unknown Tall": {"class_name": "Unknown Tall Coconut", "lifespan": "Unknown",
                     "definition": "Possibly from a tall coconut group."},
    "Unknown Dwarf": {"class_name": "Unknown Dwarf Coconut", "lifespan": "Unknown",
                      "definition": "Possibly from a dwarf coconut group."},
    "NotCoconut": {"class_name": "Invalid Image", "lifespan": "None", "definition": "None"}
}

# ---------------- HELPER FUNCTIONS ----------------
model = None

def lazy_load_model():
    global model
    if model is None:
        print("📦 Loading CNN model...")
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    return model

def pil_from_base64(data_base64: str):
    if data_base64.startswith("data:"):
        data_base64 = data_base64.split(",", 1)[1]
    decoded = base64.b64decode(data_base64)
    return Image.open(io.BytesIO(decoded)).convert("RGB")

def pil_from_file_storage(file_storage):
    return Image.open(io.BytesIO(file_storage.read())).convert("RGB")

def preprocess_pil_image(pil_img):
    pil_img = pil_img.resize(IMG_SIZE)
    arr = np.array(pil_img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict_coconut_from_pil(pil_img):
    mdl = lazy_load_model()
    img_array = preprocess_pil_image(pil_img)
    preds = mdl.predict(img_array, verbose=0)
    pred_index = int(np.argmax(preds))
    pred_conf = float(preds[0][pred_index])
    pred_class = CLASSES[pred_index]

    # Handle unknown/low confidence
    if pred_class in ["Unknown Dwarf Variety", "Unknown Tall Variety"]:
        pred_class = "Unknown Dwarf" if "Dwarf" in pred_class else "Unknown Tall"
        pred_conf = 0.55
    elif pred_class == "NotCoconut" or pred_conf < CONFIDENCE_THRESHOLD:
        pred_class = "NotCoconut"
        pred_conf = 0.0

    info = CLASS_INFO.get(pred_class, {})
    is_valid = pred_class not in ["Unknown Dwarf", "Unknown Tall", "NotCoconut"]

    return {
        "class_name": info.get("class_name", pred_class),
        "lifespan": info.get("lifespan", "N/A"),
        "definition": info.get("definition", "N/A"),
        "confidence": round(pred_conf, 4),
        "is_valid": is_valid
    }

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    pil_img = None
    image_source = None

    if request.is_json:
        data = request.get_json(silent=True) or {}
        if "image_base64" in data:
            pil_img = pil_from_base64(data["image_base64"])
            image_source = "base64"

    if pil_img is None and "image" in request.files:
        pil_img = pil_from_file_storage(request.files["image"])
        image_source = "multipart"

    if pil_img is None:
        return jsonify({"error": "No image provided"}), 400

    result = predict_coconut_from_pil(pil_img)
    result["image_source"] = image_source

    # Save to Firestore if valid
    if db and result.get("is_valid", False):
        try:
            db.collection("CoconutPredictions").add(result)
        except Exception as e:
            print(f"⚠️ Firebase save failed: {e}")

    return jsonify(result)

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 Server running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)

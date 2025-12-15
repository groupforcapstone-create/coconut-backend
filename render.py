# ============================================================
#                       render.py
# ============================================================

import os
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import firebase_admin
from firebase_admin import credentials, firestore

# ---------------- FLASK SETUP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))
CORS(app)

# ---------------- FIREBASE SETUP ----------------
db = None
CLASS_INFO = {
    "Baybay Tall Coconut": {
        "class_name": "Baybay Tall Coconut",
        "lifespan": "60-90 years",
        "definition": "A tall coconut variety commonly grown for its strong trunk and high yield."
    },
    "Catigan Dwarf Coconut": {
        "class_name": "Catigan Dwarf Coconut",
        "lifespan": "60-90 years",
        "definition": "A dwarf coconut variety known for early fruiting and consistent nut production."
    },
    "Laguna Tall Coconut": {
        "class_name": "Laguna Tall Coconut",
        "lifespan": "60-90 years",
        "definition": "A tall coconut variety recognized for its durability and adaptability."
    },
    "Tacunan Dwarf Coconut": {
        "class_name": "Tacunan Dwarf Coconut",
        "lifespan": "60-90 years",
        "definition": "A compact dwarf coconut variety valued for its high-quality nuts."
    },
    "Unknown Tall": {
        "class_name": "Unknown Tall Coconut",
        "lifespan": "Unknown",
        "definition": "Possibly from a tall coconut group."
    },
    "Unknown Dwarf": {
        "class_name": "Unknown Dwarf Coconut",
        "lifespan": "Unknown",
        "definition": "Possibly from a dwarf coconut group."
    },
    "NotCoconut": {
        "class_name": "Invalid Image",
        "lifespan": "None",
        "definition": "None"
    }
}

try:
    # Use JSON from ENV VAR (Render Secret)
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
    print("⚠️ Firebase initialization failed:", e)

# ---------------- ROUTES ----------------
@app.route("/")
@app.route("/index.html")
def index_html():
    return render_template("index.html")

@app.route("/dashboard.html")
def dashboard_html():
    return render_template("dashboard.html")

@app.route("/admin.html")
def admin_html():
    return render_template("admin.html")

@app.route("/register.html")
def register_html():
    return render_template("register.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON:
    {
        "class_name": "Baybay Tall Coconut",
        "location": "Laguna"
    }
    Saves valid predictions to Firestore.
    """
    if not db:
        return jsonify({"error": "Firestore not initialized"}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    class_name = data.get("class_name", "Unknown")
    location = data.get("location", "Unknown")

    info = CLASS_INFO.get(class_name, {
        "class_name": class_name,
        "lifespan": "Unknown",
        "definition": "No info available"
    })

    is_valid = class_name in CLASS_INFO and class_name not in ["Unknown Tall", "Unknown Dwarf", "NotCoconut"]

    result = {
        "class_name": info["class_name"],
        "lifespan": info["lifespan"],
        "definition": info["definition"],
        "location": location,
        "confidence": 1.0 if is_valid else 0.0,
        "is_valid": is_valid
    }

    # Save to Firestore if valid
    if db and is_valid:
        try:
            db.collection("CoconutPredictions").add(result)
        except Exception as e:
            print("⚠️ Firestore write failed:", e)

    return jsonify(result)

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Firestore + Pages backend running on port {port}")
    app.run(host="0.0.0.0", port=port)

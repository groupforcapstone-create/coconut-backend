import os
import cv2
import numpy as np
import tensorflow as tf
import mysql.connector
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- CLOUD CONFIGURATION ---
MODEL_PATH = "coconut_model_v2_ultra.h5" 
IMG_SIZE = (224, 224)
# Siguraduhin na ang CLASS_NAMES ay tugma sa pagkakasunod-sunod ng model mo
CLASS_NAMES = ["Baybay Tall Coconut", "Catigan Dwarf Coconut", "NotCoconut", "Tacunan Dwarf Coconut"]

# Remote Database Config
db_config = {
    "host": "148.222.53.5",
    "user": "u914267632_group4",
    "password": "Wowgaling@12345",
    "database": "u914267632_coconutproject", 
    "port": 3306,
    "connect_timeout": 10 
}

# --- LOAD AI MODEL ---
def load_model_file():
    print("⏳ Loading AI Model... please wait.")
    try:
        # load_model with compile=False for faster loading
        loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully!")
        return loaded_model
    except Exception as e:
        print(f"❌ Model Load Error: {e}")
        return None

model = load_model_file()

# --- DATABASE LOGIC ---
def save_to_db(variety, confidence, address):
    print(f"💾 Attempting to save to Remote DB: {variety}...")
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        query = """INSERT INTO detections (variety_name, confidence, address, created_at, updated_at) 
                   VALUES (%s, %s, %s, NOW(), NOW())"""
        cursor.execute(query, (variety, confidence, address))
        connection.commit()
        cursor.close()
        connection.close()
        print("✅ Database sync complete.")
    except Exception as e:
        print(f"❌ Database Error: {e}")

# --- API ROUTES ---

@app.route("/", methods=["GET"])
def health_check():
    print("👋 Health check requested by device.")
    return jsonify({"status": "Server is Live", "database": "Remote Connected"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    print("\n🔔 RECEIVED SCAN REQUEST")
    
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    if 'file' not in request.files:
        print("⚠️ No file found in request.")
        return jsonify({"error": "No image sent"}), 400
    
    address = request.form.get('address', 'Unknown Location')
    print(f"📍 Location: {address}")

    try:
        # 1. Process Image
        file = request.files['file']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE)
        img_final = np.expand_dims(img_resized.astype("float32") / 255.0, axis=0)

        # 2. Run Prediction
        print("🧠 Running AI Inference...")
        preds = model.predict(img_final, verbose=0)[0]
        
        # 3. Create List of Predictions for Flutter's Progress Bars
        top_predictions = []
        for i in range(len(CLASS_NAMES)):
            top_predictions.append({
                "label": CLASS_NAMES[i],
                "confidence": float(preds[i]) # Flutter will convert this to percentage
            })

        # Sort predictions: pinakamataas ang nauuna
        top_predictions.sort(key=lambda x: x['confidence'], reverse=True)

        idx = np.argmax(preds)
        label = CLASS_NAMES[idx]
        confidence = float(preds[idx]) * 100
        
        print(f"🎯 Top Result: {label} ({confidence:.2f}%)")

        # 4. Handle "Not a Coconut" Logic
        if label == "NotCoconut":
            print("🚫 Result is NotCoconut. Skipping DB save.")
            return jsonify({
                "variety_name": "Not a Coconut",
                "confidence": confidence,
                "address": address,
                "top_predictions": top_predictions, # Ipadala pa rin para sa UI
                "lifespan": "N/A",
                "definition": "The object scanned does not match any known coconut seedling varieties."
            })

        # 5. Save to DB only if it's a Coconut
        save_to_db(label, confidence, address)

        # 6. Return Data to Flutter
        return jsonify({
            "variety_name": label,
            "confidence": round(confidence, 2),
            "address": address,
            "top_predictions": top_predictions,
            "lifespan": "60-80 years", # Pwede mo 'tong gawing dynamic base sa variety
            "definition": f"This is a healthy {label} seedling ready for planting."
        })

    except Exception as e:
        print(f"🔥 Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Flask Server starting on http://0.0.0.0:8001")
    app.run(host='0.0.0.0', port=8001, debug=False)
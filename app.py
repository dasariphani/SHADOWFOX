from flask import Flask, request, jsonify
import os
import joblib
import numpy as np

# Create Flask app (IMPORTANT: must be named 'app')
app = Flask(__name__)

# Load model safely (works on Vercel)
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print("⚠️ Model file not found")

# Home route
@app.route("/")
def home():
    return "Car Price Predictor API is running 🚗"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Example: expecting features like [year, km_driven, fuel_type...]
        features = data.get("features")

        if not features:
            return jsonify({"error": "No features provided"}), 400

        # Convert to numpy array
        input_data = np.array(features).reshape(1, -1)

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        prediction = model.predict(input_data)

        return jsonify({
            "prediction": float(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500
from flask import Flask, request, jsonify, render_template
import os
import joblib
import numpy as np

app = Flask(__name__)

# Ensure the model path is correct
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "car_price_model.pkl")

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print("⚠️ Model file not found")

@app.route("/")
def home():
    # This line tells Flask to look for index.html in the 'templates' folder
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = data.get("features")
        if not features or model is None:
            return jsonify({"error": "Data or Model missing"}), 400
        
        input_data = np.array(features).reshape(1, -1)
        prediction = model.predict(input_data)
        return jsonify({"prediction": round(float(prediction[0]), 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
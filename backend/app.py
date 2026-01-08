from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import joblib
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")
MODEL_DIR = os.path.join(BASE_DIR, "../models")

# ---------------- LOAD ML MODEL ----------------
traffic_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "traffic_lstm.keras")
)
scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)

# ---------------- PAGE ROUTES ----------------

@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "home.html")

@app.route("/predict-page")
def predict_page():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/working")
def working():
    return send_from_directory(FRONTEND_DIR, "working.html")

@app.route("/model")
def model_page():   # ✅ RENAMED (IMPORTANT)
    return send_from_directory(FRONTEND_DIR, "model.html")

@app.route("/script.js")
def script():
    return send_from_directory(FRONTEND_DIR, "script.js")

# ---------------- API ROUTE ----------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        traffic_volume = float(data["traffic_volume"])
        hour = int(data["hour"])
        day = int(data["day"])

        # Cyclic encoding
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day / 7)
        day_cos = np.cos(2 * np.pi * day / 7)

        features = np.array([[latitude, longitude, traffic_volume,
                              hour_sin, hour_cos, day_sin, day_cos]])

        scaled = scaler.transform(features)
        reshaped = scaled.reshape((1, 1, scaled.shape[1]))

        prediction = float(traffic_model.predict(reshaped)[0][0])

        risk_level = (
            "Low" if prediction < 0.3 else
            "Medium" if prediction < 0.7 else
            "High"
        )

        return jsonify({
            "accident_risk": round(prediction, 3),
            "risk_level": risk_level
        })

    except Exception as e:
        print("PREDICTION ERROR:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# =========================
# Load ML artifacts
# =========================
model = joblib.load("artifacts/models/model.pkl")
scaler = joblib.load("artifacts/processed/scaler.pkl")

# =========================
# OpenRouter API Key
# =========================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

FEATURES = [
    'Operation_Mode', 'Temperature_C', 'Vibration_Hz',
    'Power_Consumption_kW', 'Network_Latency_ms', 'Packet_Loss_%',
    'Quality_Control_Defect_Rate_%', 'Production_Speed_units_per_hr',
    'Predictive_Maintenance_Score', 'Error_Rate_%',
    'Year', 'Month', 'Day', 'Hour'
]

LABELS = {
    0: "High",
    1: "Low",
    2: "Medium"
}

# =========================
# HOME ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    input_values = None

    if request.method == "POST":
        try:
            input_data = [float(request.form[feature]) for feature in FEATURES]
            input_values = dict(zip(FEATURES, input_data))

            X = np.array(input_data).reshape(1, -1)
            X = scaler.transform(X)

            pred = model.predict(X)[0]
            prediction = LABELS.get(pred, "Unknown")

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        features=FEATURES,
        input_values=input_values
    )

# =========================
# LLM EXPLANATION ENDPOINT
# =========================
@app.route("/explain", methods=["POST"])
def explain():
    try:
        data = request.json

        prediction = data.get("prediction")

        prompt = f"""
You are an industrial AI operations expert.

A machine learning model classified a machine as:

Classification: {prediction}

Task:
1. Explain what this classification means in simple terms
2. Possible operational implications
3. Recommendations for maintenance or optimization
4. Urgency level (Low / Medium / High)

Do NOT assume missing sensor data. Only interpret the classification.
"""

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "inclusionai/ring-2.6-1t:free",
                "messages": [
                    {"role": "system", "content": "You are an industrial AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4
            },
            timeout=20
        )

        result = response.json()

        if "choices" not in result:
            return jsonify({"error": result})

        return jsonify({
            "insight": result["choices"][0]["message"]["content"]
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
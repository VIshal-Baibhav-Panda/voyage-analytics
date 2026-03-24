from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask import render_template

app = Flask(__name__)

# ✅ CORRECT PATHS (NO ../)
import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "../models/flight_price_model.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "../models/encoders.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "../models/columns.pkl"))

# ✅ HOME ROUTE (VISIBLE IN BROWSER)
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict-price", methods=["POST"])
def predict_price():
    try:
        data = request.json
        input_data = []

        for col in columns:
            if col not in data:
                return jsonify({"error": f"Missing field: {col}"})

            value = data[col]

            # Encode safely
            if col in encoders:
                if value not in encoders[col].classes_:
                    return jsonify({"error": f"Invalid value '{value}' for {col}"})
                value = encoders[col].transform([value])[0]

            input_data.append(value)

        prediction = model.predict(np.array(input_data).reshape(1, -1))

        return jsonify({"price": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route("/predict-gender", methods=["POST"])
def gender():
    data = request.json
    user_code = data.get("userCode")

    result = predict_gender(user_code)

    return jsonify({
        "gender": result
    })

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json

    source = data.get("from")
    budget = data.get("budget", 5000)

    result = recommend_destination(source, budget)

    return jsonify({
        "recommendations": result
    })
    
# ✅ DOCKER FIX
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
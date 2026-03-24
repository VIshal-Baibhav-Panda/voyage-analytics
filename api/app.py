from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# ✅ CORRECT PATHS (NO ../)
model = joblib.load("models/flight_price_model.pkl")
encoders = joblib.load("models/encoders.pkl")
columns = joblib.load("models/columns.pkl")

# ✅ HOME ROUTE (VISIBLE IN BROWSER)
@app.route("/")
def home():
    return "<h1>API Running ✅</h1>"

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
# ✅ DOCKER FIX
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
import streamlit as st
import requests
import joblib

st.title("✈️ Flight Price Predictor")

columns = joblib.load("../models/columns.pkl")

data = {}

for col in columns:
    if col in ["time", "distance", "userCode"]:
        data[col] = st.number_input(col, min_value=0)
    else:
        data[col] = st.text_input(col)

if st.button("Predict Price"):
    try:
        response = requests.post(
            "http://127.0.0.1:5001/predict-price",
            json=data,
            timeout=5
        )

        result = response.json()

        if result["status"] == "success":
            st.success(f"💰 Price: ₹{round(result['predicted_price'],2)}")
        else:
            st.error(result["message"])

    except Exception as e:
        st.error(f"Error: {e}")
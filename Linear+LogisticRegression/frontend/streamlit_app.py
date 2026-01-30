import streamlit as st
import requests

st.set_page_config(page_title="House ML App", layout="centered")

st.title("🏠 House Price & Sale Prediction")

st.write("Enter house details:")

area = st.number_input("Area (sqft)", 500, 10000, 2500)
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.number_input("Bathrooms", 1, 10, 2)
floors = st.number_input("Floors", 1, 5, 2)
age = st.number_input("Age (years)", 0, 100, 10)
location = st.slider("Location Score", 1.0, 10.0, 7.5)

if st.button("Predict"):
    payload = {
        "area_sqft": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floors": floors,
        "age_years": age,
        "location_score": location
    }

    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json=payload,
            timeout=5
        )

        if response.status_code == 200:
            result = response.json()

            st.success(f"💰 Predicted Price: ₹{result['predicted_price']}")
            st.info(f"📊 Sold Probability: {result['sold_probability']}")

            if result["sold_within_week"]:
                st.success("🔥 Likely to sell within a week")
            else:
                st.warning("⏳ May take longer to sell")

        else:
            st.error("❌ Backend error")

    except requests.exceptions.ConnectionError:
        st.error("❌ Flask backend is not running. Start backend first.")

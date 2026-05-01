import streamlit as st

st.title("🌦️ Weather Prediction App")

temp = st.slider("Temperature (°C)", -10, 40, 25)
wind = st.slider("Wind Speed (km/h)", 0, 50, 10)
pressure = st.slider("Pressure", 900, 1100, 1010)

if temp > 30:
    risk = "High Risk"
elif temp > 15:
    risk = "Medium Risk"
else:
    risk = "Low Risk"

st.subheader(f"Prediction: {risk}")

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# ================= UI SETTINGS =================
st.set_page_config(page_title="Weather AI", layout="centered")

# Custom CSS
st.markdown("""
<style>
.card {
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 10px;
    text-align: center;
    font-size: 20px;
    font-weight: bold;
}
.low {background-color: #d4edda; color: #155724;}
.medium {background-color: #fff3cd; color: #856404;}
.high {background-color: #f8d7da; color: #721c24;}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("🌦️ Weather Severity Prediction AI")
st.write("AI system to predict weather safety and explain decisions")

# ================= MODEL INFO =================
st.sidebar.header("📊 Model Info")
st.sidebar.write("Algorithm: Random Forest Classifier")
st.sidebar.write("Trained on weather dataset")

# ================= INPUT =================
temp = st.number_input("🌡️ Temperature (°C)", value=25.0)
wind_speed = st.number_input("💨 Wind Speed (km/h)", value=10.0)
pressure = st.number_input("🌍 Pressure (millibars)", value=1010.0)

# Load model
model = joblib.load("model.pkl")

# ================= BUTTON =================
if st.button("🔍 Predict Weather Risk"):

    data = pd.DataFrame({
        'Temperature (C)': [temp],
        'Wind Speed (km/h)': [wind_speed],
        'Pressure (millibars)': [pressure]
    })

    prediction = model.predict(data)
    risk = prediction[0]

    # ================= CONFIDENCE =================
    try:
        probs = model.predict_proba(data)
        confidence = np.max(probs) * 100
        st.info(f"📊 Prediction Confidence: **{confidence:.2f}%**")
    except:
        st.info("📊 Confidence not available")

    # ================= RISK CARD =================
    st.subheader("⚠️ Weather Severity Level")

    if risk == "Low Risk":
        st.markdown('<div class="card low">🟢 LOW RISK<br>Safe Weather</div>', unsafe_allow_html=True)
    elif risk == "Medium Risk":
        st.markdown('<div class="card medium">🟡 MEDIUM RISK<br>Moderate Conditions</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="card high">🔴 HIGH RISK<br>Dangerous Weather</div>', unsafe_allow_html=True)

    # ================= WEATHER TYPE =================
    if temp > 30 and wind_speed < 15:
        weather = "☀️ Sunny"
    elif wind_speed > 30:
        weather = "💨 Windy"
    elif pressure < 1000:
        weather = "🌧️ Rainy"
    else:
        weather = "🌤️ Moderate"

    st.subheader("🌦️ Weather Condition")
    st.info(weather)

    # ================= PRECAUTIONS =================
    st.subheader("⚠️ Precautions")

    if risk == "Low Risk":
        precautions = [
            "Wear sunscreen if outdoors for extended periods",
            "Stay hydrated, especially in hot weather",
            "Carry a light jacket for cooler evenings"
        ]
    elif risk == "Medium Risk":
        precautions = [
            "Carry an umbrella or raincoat",
            "Avoid long exposure to wind or sun",
            "Monitor weather updates regularly",
            "Wear comfortable, layered clothing"
        ]
    else:  # High Risk
        precautions = [
            "Stay indoors if possible",
            "Avoid travel unless necessary",
            "Prepare emergency supplies",
            "Follow local weather alerts",
            "Secure outdoor items"
        ]

    for precaution in precautions:
        st.write(f"• {precaution}")

    # ================= SHAP =================
    import shap

    st.subheader("🧠 Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)

    if isinstance(shap_values, list):
        shap_df = pd.DataFrame(shap_values[0], columns=data.columns)
    elif len(shap_values.shape) == 3:
        shap_df = pd.DataFrame(shap_values[:, :, 0], columns=data.columns)
    else:
        shap_df = pd.DataFrame(shap_values, columns=data.columns)

    fig, ax = plt.subplots()
    shap_values_row = shap_df.iloc[0].values
    feature_names = shap_df.columns

    colors = ['green' if x > 0 else 'red' for x in shap_values_row]

    ax.barh(feature_names, np.abs(shap_values_row), color=colors)
    ax.set_title("Feature Impact")

    st.pyplot(fig)

    # ================= AI EXPLANATION =================
    st.subheader("🤖 AI Insight")

    increase = []
    decrease = []

    for i, val in enumerate(shap_values_row):
        if val > 0:
            increase.append(feature_names[i])
        else:
            decrease.append(feature_names[i])

    st.info(
        f"""
        Prediction: **{risk}**

        🔺 Increased severity: {", ".join(increase) if increase else "None"}  
        🔻 Reduced severity: {", ".join(decrease) if decrease else "None"}  

        🌦️ Final condition: {weather}
        """
    )
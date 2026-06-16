import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide")

# ===============================
# LOAD MODEL + SCALER
# ===============================
@st.cache_resource
def load_assets():
    model = joblib.load("models/fraud_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_assets()

# ===============================
# FEATURE ORDER (VERY IMPORTANT)
# ===============================
FEATURES = [
    'V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
    'V11','V12','V13','V14','V15','V16','V17','V18',
    'V19','V20','V21','V22','V23','V24','V25','V26',
    'V27','V28','Hour','Log_Amount'
]

# ===============================
# RISK SCORE FUNCTION
# ===============================
def fraud_risk(model, X):
    prob = model.predict_proba(X)[0][1]
    score = round(prob * 100, 2)

    if score >= 75:
        level = "HIGH RISK 🔴"
    elif score >= 40:
        level = "MEDIUM RISK 🟡"
    else:
        level = "LOW RISK 🟢"

    return score, level, prob

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🛡️ Fraud Detection System")
page = st.sidebar.radio("Navigation", ["Home", "Predict", "Model Info"])

# ===============================
# HOME PAGE
# ===============================
if page == "Home":
    st.title("💳 Fraud Transaction Detection")
    st.write("Machine Learning powered fraud detection system using XGBoost.")

    st.success("Model Ready for Predictions 🚀")

# ===============================
# PREDICTION PAGE
# ===============================
elif page == "Predict":

    st.title("🔍 Transaction Analysis")

    amount = st.number_input("Transaction Amount", value=100.0)
    hour = st.slider("Hour of Transaction", 0, 23, 12)

    st.subheader("Enter PCA Features (V1 - V28)")

    cols = st.columns(4)

    v = []
    for i in range(28):
        v.append(cols[i % 4].number_input(f"V{i+1}", value=0.0))

    if st.button("Predict Fraud"):

        # ===============================
        # CREATE INPUT DATAFRAME (FIXED)
        # ===============================
        input_data = v + [hour, np.log1p(amount)]

        X_input = pd.DataFrame([input_data], columns=FEATURES)

        # ===============================
        # PREDICTION
        # ===============================
        pred = model.predict(X_input)[0]
        score, level, prob = fraud_risk(model, X_input)

        st.markdown("---")

        if pred == 1:
            st.error("⚠️ FRAUD DETECTED")
        else:
            st.success("✅ LEGITIMATE TRANSACTION")

        st.metric("Fraud Probability", f"{prob*100:.2f}%")
        st.metric("Risk Score", f"{score}/100")
        st.info(level)

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Fraud Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "red"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

# ===============================
# MODEL INFO PAGE
# ===============================
elif page == "Model Info":

    st.title("📊 Model Information")

    st.write("Best Model: XGBoost")
    st.write("Trained on Credit Card Fraud Dataset")

    st.success("Handles extreme class imbalance using SMOTE + class weighting")
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# ---------------------------
# Load Model
# ---------------------------
MODEL_PATH = "final_logreg_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found. Please upload it to the same folder as this app.")
    st.stop()

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Bankruptcy Prediction App", page_icon="🏦", layout="centre")

# ---------------------------
# App Header
# ---------------------------
st.title("🏦 Bankruptcy Prediction App")
st.write(
    """
    This app uses a **Logistic Regression Model** to predict whether a company 
    is likely to go **Bankrupt** or remain **Financially Healthy** based on its financial indicators.  
    Select values (0, 0.5, or 1) for each feature below and click **Predict**.
    """
)

# ---------------------------
# Helper Function
# ---------------------------
def prepare_features(df, model):
    """Ensure columns match the trained model."""
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        for col in expected:
            if col not in df.columns:
                df[col] = 0.0
        X = df.reindex(columns=expected)
    else:
        X = df.select_dtypes(include=[np.number]).copy()
    return X.fillna(0)

# ---------------------------
# Manual Input Section
# ---------------------------
st.header("📊 Enter Company Financial Indicators")

options = [0.0, 0.5, 1.0]
col1, col2 = st.columns(2)

with col1:
    industrial_risk = st.selectbox("Industrial Risk", options, index=1)
    management_risk = st.selectbox("Management Risk", options, index=1)
    financial_flexibility = st.selectbox("Financial Flexibility", options, index=1)

with col2:
    credibility = st.selectbox("Credibility", options, index=1)
    competitiveness = st.selectbox("Competitiveness", options, index=1)
    operating_risk = st.selectbox("Operating Risk", options, index=1)

input_data = pd.DataFrame([{
    "industrial_risk": industrial_risk,
    "management_risk": management_risk,
    "financial_flexibility": financial_flexibility,
    "credibility": credibility,
    "competitiveness": competitiveness,
    "operating_risk": operating_risk
}])

# ---------------------------
# Predict Button
# ---------------------------
if st.button("🔍 Predict Bankruptcy"):
    try:
        X = prepare_features(input_data, model)
        prediction = model.predict(X)[0]

        # Probabilities
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            prob_bankruptcy = probs[0]
            prob_non_bankruptcy = probs[1]
        else:
            prob_bankruptcy = 1.0 if prediction == 0 else 0.0
            prob_non_bankruptcy = 1.0 - prob_bankruptcy

        # Display Results
        if prediction == 0:
            st.error("⚠️ The company is **at risk of Bankruptcy**. Consider improving liquidity and reducing risk.")
        else:
            st.success("✅ The company is **Financially Healthy**. Maintain good financial practices and management.")

        # Metrics Display
        st.subheader("📈 Predicted Probabilities")
        colA, colB = st.columns(2)
        colA.metric("Bankruptcy Probability", f"{prob_bankruptcy*100:.1f}%")
        colB.metric("Non-Bankruptcy Probability", f"{prob_non_bankruptcy*100:.1f}%")

        # Donut Chart
        fig = go.Figure(data=[go.Pie(
            labels=["Bankruptcy", "Non-Bankruptcy"],
            values=[prob_bankruptcy, prob_non_bankruptcy],
            hole=0.5,
            marker=dict(colors=["#ef553b", "#00cc96"]),
            textinfo="percent+label",
        )])
        fig.update_layout(margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")


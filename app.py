import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from time import sleep

# ---------------------------
# Page config (must come first)
# ---------------------------
st.set_page_config(page_title="🏦 Bankruptcy Predictor", page_icon="💰", layout="wide")

# ---------------------------
# CSS with Adaptive Theming
# ---------------------------
st.markdown("""
<style>
/* Use Streamlit theme variables for auto light/dark adaptation */
:root {
  --primary: var(--primary-color);
  --bg: var(--background-color);
  --text: var(--text-color);
}

/* Base styles */
html, body, [data-testid="stAppViewContainer"] {
  color: var(--text);
  background: var(--bg);
  font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
  transition: background 0.3s, color 0.3s;
}

/* Gradient header */
.hero {
  background: linear-gradient(90deg, rgba(0,119,255,0.12), rgba(0,194,168,0.08));
  border-radius: 16px;
  padding: 22px 28px;
  display: flex;
  align-items: center;
  gap: 18px;
  box-shadow: 0 8px 30px rgba(0,0,0,0.05);
  border: 1px solid rgba(128,128,128,0.1);
}
.hero h1 {
  font-size: 30px;
  font-weight: 800;
  margin: 0;
}
.hero p {
  margin: 0;
  font-size: 14px;
  opacity: 0.8;
}

/* Glass card */
.glass {
  background: rgba(255,255,255,0.8);
  border-radius: 12px;
  padding: 18px;
  margin-bottom: 16px;
  border: 1px solid rgba(0,0,0,0.05);
  box-shadow: 0 6px 25px rgba(0,0,0,0.06);
  backdrop-filter: blur(12px);
}
[data-theme="dark"] .glass {
  background: rgba(30,30,30,0.6);
  border: 1px solid rgba(255,255,255,0.05);
  box-shadow: 0 6px 30px rgba(0,0,0,0.4);
}

/* Button glow */
.glow-button .stButton>button {
  background: linear-gradient(90deg,#0077ff,#00c2a8)!important;
  color: #fff !important;
  border-radius: 12px !important;
  padding: 10px 18px !important;
  font-weight: 700 !important;
  transition: all 0.2s ease-in-out;
}
.glow-button .stButton>button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(0,119,255,0.25);
}

/* Metric text */
.big-metric { font-size:28px; font-weight:800; margin-bottom:3px; }
.muted { opacity:0.7; font-size:13px; }

/* Suggestion pill */
.suggest {
  padding: 10px 12px;
  border-radius: 10px;
  font-weight: 600;
  margin-top: 10px;
  background: rgba(0,194,168,0.08);
  color: var(--text);
  border-left: 5px solid #00c2a8;
}

/* Footer */
.footer { text-align:center; opacity:0.7; margin-top:20px; font-size:13px; }

</style>
""", unsafe_allow_html=True)

# ---------------------------
# Model loading
# ---------------------------
MODEL_PATH = "final_logreg_model.pkl"

@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error("❌ Model file not found. Please upload `final_logreg_model.pkl` to this app folder.")
    st.stop()

# ---------------------------
# Helper Functions
# ---------------------------
def prepare_features(df, model):
    X = df.copy()
    for col in ["class", "target", "label"]:
        if col in X.columns:
            X = X.drop(columns=[col])
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        for col in expected:
            if col not in X.columns:
                X[col] = 0.0
        X = X.reindex(columns=expected)
    else:
        X = X.select_dtypes(include=[np.number])
    return X.fillna(0)

def get_class_indices(model):
    classes = list(getattr(model, "classes_", []))
    idx_bank = 0
    try:
        idx_bank = classes.index(0)
    except ValueError:
        pass
    idx_non = 1 if len(classes) > 1 else 0
    return idx_bank, idx_non

def fancy_donut(prob_bank, prob_non):
    fig = go.Figure(data=[go.Pie(
        labels=["Bankruptcy", "Non-Bankruptcy"],
        values=[prob_bank, prob_non],
        hole=0.6,
        marker=dict(colors=["#ff6b6b", "#20c997"]),
        hoverinfo='label+percent',
        textinfo='none'
    )])
    fig.update_layout(
        showlegend=True,
        annotations=[
            dict(
                text=f"<b>{prob_bank*100:.1f}%</b><br><span style='font-size:12px'>Bankrupt</span><br>"
                     f"<b>{prob_non*100:.1f}%</b><br><span style='font-size:12px'>Healthy</span>",
                x=0.5, y=0.5, showarrow=False
            )
        ],
        margin=dict(t=10, b=10, l=10, r=10)
    )
    return fig

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.title("ℹ️ About the App")
    st.write("Predict company **bankruptcy risk** using a trained Logistic Regression model.")
    st.markdown("---")
    uploaded = st.file_uploader("📂 Upload CSV/XLSX for batch prediction", type=["csv", "xlsx"])
    st.caption("Upload a dataset containing the 6 feature columns.")
    st.markdown("---")
    st.write("Model Type:", model.__class__.__name__)

# ---------------------------
# Hero Header
# ---------------------------
st.markdown("""
<div class="hero">
  <div>
    <h1>🏦 Bankruptcy Prediction App</h1>
    <p>Predict company financial health — automatically adjusts visuals for dark & light themes.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Single Prediction Section
# ---------------------------
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("📊 Single Prediction")

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

input_df = pd.DataFrame([{
    "industrial_risk": industrial_risk,
    "management_risk": management_risk,
    "financial_flexibility": financial_flexibility,
    "credibility": credibility,
    "competitiveness": competitiveness,
    "operating_risk": operating_risk
}])

st.markdown('<div class="glow-button">', unsafe_allow_html=True)
predict = st.button("🔍 Predict Bankruptcy")
st.markdown('</div>', unsafe_allow_html=True)

if predict:
    with st.spinner("Analyzing company data..."):
        sleep(0.6)
        try:
            X = prepare_features(input_df, model)
            pred = model.predict(X)[0]
            probs = model.predict_proba(X)[0]
            idx_bank, idx_non = get_class_indices(model)
            prob_bank, prob_non = probs[idx_bank], probs[idx_non]

            c1, c2 = st.columns([1, 1])
            with c1:
                if pred == 0:
                    st.error("⚠️ **Prediction:** RISK OF BANKRUPTCY")
                    st.markdown('<div class="suggest">💡 Suggestion: Improve cash flow and management efficiency.</div>', unsafe_allow_html=True)
                else:
                    st.success("✅ **Prediction:** FINANCIALLY HEALTHY")
                    st.markdown('<div class="suggest">💡 Suggestion: Maintain competitive advantage and credit health.</div>', unsafe_allow_html=True)
                st.markdown("<div class='big-metric'>"
                            f"{prob_bank*100:.1f}% <span class='muted'>Bankruptcy</span></div>", unsafe_allow_html=True)
                st.markdown("<div class='big-metric'>"
                            f"{prob_non*100:.1f}% <span class='muted'>Non-Bankruptcy</span></div>", unsafe_allow_html=True)
            with c2:
                st.plotly_chart(fancy_donut(prob_bank, prob_non), use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Batch Predictions (simple summary)
# ---------------------------
if uploaded is not None:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📁 Batch Dataset Prediction")
    try:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        st.write("Preview:")
        st.dataframe(df.head())

        X = prepare_features(df, model)
        preds = model.predict(X)
        df["Prediction"] = np.where(preds == 0, "Bankruptcy", "Non-Bankruptcy")

        st.success("✅ Predictions Complete")
        bankrupt_count = (df["Prediction"] == "Bankruptcy").sum()
        non_bankrupt_count = (df["Prediction"] == "Non-Bankruptcy").sum()

        st.metric("Bankruptcy", bankrupt_count)
        st.metric("Non-Bankruptcy", non_bankrupt_count)

        fig = go.Figure(data=[go.Pie(
            labels=["Bankruptcy", "Non-Bankruptcy"],
            values=[bankrupt_count, non_bankrupt_count],
            hole=0.45,
            marker=dict(colors=["#ff6b6b", "#20c997"]),
            textinfo="label+percent"
        )])
        fig.update_layout(title_text="📈 Dataset Bankruptcy Distribution", title_x=0.5)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing file: {e}")

    st.markdown('</div>', unsafe_allow_html=True)


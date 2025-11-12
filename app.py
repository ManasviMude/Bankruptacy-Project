# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import plotly.graph_objects as go

# ---------------------------
# Load model (ensure final_logreg_model.pkl is present)
# ---------------------------
with open("final_logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Bankruptcy Prediction",
    page_icon="🏦",
    layout="wide",
)

# ---------------------------
# Strong, clean WHITE + BLUE theme CSS (overrides)
# ---------------------------
st.markdown(
    """
    <style>
    /* Base */
    html, body, .main {
        background-color: #f8fbff !important;  /* very light blue */
        color: #083060 !important;              /* dark navy text */
        font-family: Inter, 'Segoe UI', Roboto, sans-serif;
    }

    /* App container card style */
    .app-card {
        background: #ffffff !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 8px 30px rgba(10, 40, 90, 0.06) !important;
        border: 1px solid rgba(4, 32, 82, 0.03) !important;
    }

    /* Header */
    .title {
        color: #04294a !important;
        font-weight: 800;
        font-size: 26px;
    }
    .subtitle {
        color: #355d7a !important;
        margin-top: 6px;
        margin-bottom: 14px;
    }

    /* Section header */
    .section-header {
        color: #0b3a66 !important;
        font-weight: 700;
        margin-bottom: 10px;
        font-size: 15px;
    }

    /* Sidebar adjustments */
    .css-1d391kg { background: #ffffff !important; } /* sidebar container */
    .stSidebar .block-container { padding-left: 18px; padding-right: 18px; }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg,#2b8ff4,#1fb6ff) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 8px 14px !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }

    /* Selectbox / Input fields */
    div[data-baseweb="select"] > div {
        background: #f3f8ff !important;
        border-radius: 8px !important;
        border: 1px solid rgba(4, 32, 82, 0.06) !important;
        color: #083060 !important;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(180deg, #ffffff, #f6fbff) !important;
        border-radius: 10px !important;
        padding: 14px !important;
        border: 1px solid rgba(4,32,82,0.04) !important;
    }

    /* Suggestion boxes */
    .suggestion {
        border-radius: 10px;
        padding: 12px;
        font-weight: 600;
        color: #04294a;
    }
    .suggestion.good {
        background: #e9fbf3 !important;
        border-left: 6px solid #00a86b !important;
    }
    .suggestion.bad {
        background: #fff4f4 !important;
        border-left: 6px solid #ff5c5c !important;
    }

    /* Ensure all text is dark */
    h1,h2,h3,h4,h5,p,span,div,label {
        color: #04294a !important;
    }

    /* Plotly background - keep white */
    .js-plotly-plot .plotly {
        background: #ffffff !important;
    }

    /* Remove any dark-mode overlays */
    [class*="dark"], [data-testid*="dark"] {
        background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Sidebar: app info & batch upload
# ---------------------------
with st.sidebar:
    st.markdown("<div style='text-align:center;margin-bottom:8px'><img alt='logo' src='https://img.icons8.com/ios-filled/48/1fb6ff/bank.png' /></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center;color:#04294a;margin-bottom:2px;'>Bankruptcy Predictor</h3>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;color:#355d7a;margin-bottom:10px;font-size:13px;'>Logistic Regression • White & Blue Theme</div>", unsafe_allow_html=True)

    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV/XLSX for batch prediction (optional)", type=["csv", "xlsx", "xls"])
    st.markdown("Tip: CSV should contain the same feature columns used to train the model.", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Advanced")
    show_metrics = st.checkbox("Show evaluation if uploaded file contains true labels", value=True)
    st.markdown("---")
    st.markdown("### Info")
    st.markdown("• Developer: Your Name  \n• Contact: your@email.com")
    if st.button("Reset / Reload"):
        st.experimental_rerun()

# ---------------------------
# Top: header and model info
# ---------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<div class='title'>🏦 Bankruptcy Prediction App</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Predict company bankruptcy risk — fast, clear, and interpretable.</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='app-card' style='text-align:left;'>", unsafe_allow_html=True)
    st.markdown(f"**Model:** {model.__class__.__name__}")
    st.markdown(f"**Supports proba:** {'Yes' if hasattr(model,'predict_proba') else 'No'}")
    classes = getattr(model, "classes_", None)
    if classes is not None:
        st.markdown(f"**Classes:** {list(classes)}")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Main card: inputs & predict
# ---------------------------
st.markdown("<div class='app-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>📊 Enter Company Financial Indicators</div>", unsafe_allow_html=True)

options = [0.0, 0.5, 1.0]
c1, c2 = st.columns(2)
with c1:
    industrial_risk = st.selectbox("Industrial Risk", options, index=1)
    management_risk = st.selectbox("Management Risk", options, index=1)
    financial_flexibility = st.selectbox("Financial Flexibility", options, index=1)
with c2:
    credibility = st.selectbox("Credibility", options, index=1)
    competitiveness = st.selectbox("Competitiveness", options, index=1)
    operating_risk = st.selectbox("Operating Risk", options, index=1)

input_df = pd.DataFrame({
    "industrial_risk":[industrial_risk],
    "management_risk":[management_risk],
    "financial_flexibility":[financial_flexibility],
    "credibility":[credibility],
    "competitiveness":[competitiveness],
    "operating_risk":[operating_risk]
})

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

if st.button("🔍 Predict Bankruptcy"):
    pred = model.predict(input_df)[0]

    # probabilities (robust mapping)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        try:
            idx_bank = list(model.classes_).index(0)
        except Exception:
            # fallback: choose index 0 as bankruptcy
            idx_bank = 0
        idx_non = 1 if idx_bank == 0 else 0
        prob_bank = float(proba[idx_bank])
        prob_non = float(proba[idx_non]) if idx_non < len(proba) else 1.0 - prob_bank
    else:
        prob_bank = 1.0 if pred == 0 else 0.0
        prob_non = 1.0 - prob_bank

    # result message + suggestion
    if pred == 0:
        st.markdown("<div class='suggestion bad' style='margin-bottom:12px;'>⚠️ <b>Result:</b> Model predicts Bankruptcy risk. Consider reviewing liquidity & management.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='suggestion good' style='margin-bottom:12px;'>✅ <b>Result:</b> Model predicts Financially Healthy. Continue monitoring.</div>", unsafe_allow_html=True)

    # probability metric cards
    colA, colB = st.columns(2)
    with colA:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700;color:#17456b'>Bankruptcy</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:28px;font-weight:800;color:#052a44'>{prob_bank*100:.1f} %</div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#617d94;font-size:12px;margin-top:6px;'>Probability (model)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with colB:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700;color:#0b7a55'>Non-Bankruptcy</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:28px;font-weight:800;color:#052a44'>{prob_non*100:.1f} %</div>", unsafe_allow_html=True)
        st.markdown("<div style='color:#617d94;font-size:12px;margin-top:6px;'>Probability (model)</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Plotly donut + horizontal bar
    fig = go.Figure(data=[go.Pie(
        labels=["Bankruptcy", "Non-Bankruptcy"],
        values=[prob_bank, prob_non],
        hole=0.5,
        marker=dict(colors=["#ff6b6b","#2b8ff4"]),
        sort=False,
        textinfo='percent+label'
    )])
    fig.update_layout(margin=dict(l=10,r=10,t=10,b=10),
                      annotations=[dict(text='Probabilities', x=0.5, y=0.5, font_size=14, showarrow=False)])
    st.plotly_chart(fig, use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)  # close main card

# ---------------------------
# Batch upload handling (optional)
# ---------------------------
if uploaded is not None:
    st.markdown("<div class='app-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📁 Batch predictions & evaluation</div>", unsafe_allow_html=True)

    try:
        if uploaded.name.endswith((".xls", ".xlsx")):
            batch = pd.read_excel(uploaded)
        else:
            batch = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        batch = None

    if batch is not None:
        st.markdown("**Preview:**")
        st.dataframe(batch.head())

        # try to detect features
        model_features = getattr(model, "feature_names_in_", None)
        if model_features is None:
            model_features = [c for c in batch.columns if c.lower() not in ("class","target","y","label")]

        missing = [c for c in model_features if c not in batch.columns]
        if missing:
            st.warning(f"Uploaded file missing some model features: {missing}. App will attempt to use available columns.")
        X_batch = batch.reindex(columns=[c for c in model_features if c in batch.columns]).fillna(0)

        preds = model.predict(X_batch)
        batch["prediction"] = preds
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_batch)
            try:
                idx_bank = list(model.classes_).index(0)
            except Exception:
                idx_bank = 0
            batch["prob_bankruptcy"] = proba[:, idx_bank]
            batch["prob_nonbank"] = 1 - proba[:, idx_bank]
        st.success("✅ Batch predictions complete.")
        st.dataframe(batch.head())

        csv_bytes = batch.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download predictions CSV", data=csv_bytes, file_name="batch_predictions.csv", mime="text/csv")

        # evaluation if label present
        label_col = None
        for cand in ["class", "Class", "target", "Target", "y", "label", "Label"]:
            if cand in batch.columns:
                label_col = cand
                break

        if show_metrics and label_col:
            y_true = batch[label_col]
            y_pred = batch["prediction"]
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc:.3f}")
            m2.metric("Precision", f"{prec:.3f}")
            m3.metric("Recall", f"{rec:.3f}")
            m4.metric("F1", f"{f1:.3f}")

            cm = confusion_matrix(y_true, y_pred)
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=[f"Pred {i}" for i in range(cm.shape[1])],
                y=[f"True {i}" for i in range(cm.shape[0])],
                colorscale="Blues",
                showscale=True))
            fig_cm.update_layout(title="Confusion Matrix", margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_cm, use_container_width=True)

            if "prob_bankruptcy" in batch.columns:
                try:
                    y_true_bin = (y_true == 0).astype(int)
                    auc = roc_auc_score(y_true_bin, batch["prob_bankruptcy"])
                    fpr, tpr, _ = roc_curve(y_true_bin, batch["prob_bankruptcy"])
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
                    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
                    fig_roc.update_layout(title=f"ROC Curve (AUC = {auc:.3f})", xaxis_title="FPR", yaxis_title="TPR", margin=dict(l=10,r=10,t=30,b=10))
                    st.plotly_chart(fig_roc, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not compute ROC: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


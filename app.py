import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Load trained model
# ---------------------------
with open('final_logreg_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Bankruptcy Prediction App",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# Strong CSS overrides to ensure LIGHT background and DARK text everywhere
# (uses !important to override Streamlit / browser dark-mode)
# ---------------------------
st.markdown(
    """
    <style>
    /* Page background */
    html, body, .main {
        background-color: #f5f7fa !important;
        color: #111111 !important;
    }

    /* Main app container */
    .stApp, .block-container {
        background-color: #ffffff !important;
        color: #111111 !important;
    }

    /* Headings and markdown text */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #111111 !important;
    }

    /* Streamlit widgets text */
    .stText, .stMarkdown, .stMetric, .stNumberInput, .stSelectbox, .stButton {
        color: #111111 !important;
    }

    /* Buttons */
    .stButton>button {
        color: #ffffff !important;
    }

    /* Make code blocks light and text dark */
    pre, code, .stCodeBlock, .st-rtf, .media {
        background: #f3f4f6 !important;
        color: #0b0b0b !important;
        border: 1px solid rgba(0,0,0,0.06) !important;
    }

    /* Metric value style */
    .stMetricValue, .stMetricDelta {
        color: #111111 !important;
    }

    /* Tooltip / help text */
    .css-1kyxreq, .css-1v3fvcr { color: #111111 !important; }

    /* Stronger contrast for suggestion boxes */
    .suggestion-box { color: #111 !important; }

    /* Ensure pyplot text is dark */
    .stImage img, svg { color: #111111 !important; }

    /* Remove any overlay dark backgrounds */
    [class*="dark"], [data-testid*="dark"], .viewer, .code, .stExpander {
        background: transparent !important;
    }

    /* Fallback: force all text nodes inside to be dark */
    .stApp * { color: #111111 !important; }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# UI: header
# ---------------------------
st.markdown("<h1 style='text-align:center;margin-bottom:0.2rem;color:#002855;font-weight:800'>🏦 Bankruptcy Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#444;margin-top:0.2rem;margin-bottom:1.5rem;'>Predict a company's financial health using Logistic Regression</p>", unsafe_allow_html=True)

# ---------------------------
# Input section (card)
# ---------------------------
st.markdown(
    """
    <div style="background:#f0f4f8;border-radius:12px;padding:18px;margin-bottom:20px;box-shadow: inset 0 0 8px rgba(0,0,0,0.03);">
      <div style="font-size:18px;font-weight:700;color:#003366;margin-bottom:10px;">📊 Enter Company Financial Indicators</div>
    """
    , unsafe_allow_html=True
)

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

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Prepare data
# ---------------------------
data = pd.DataFrame({
    'industrial_risk': [industrial_risk],
    'management_risk': [management_risk],
    'financial_flexibility': [financial_flexibility],
    'credibility': [credibility],
    'competitiveness': [competitiveness],
    'operating_risk': [operating_risk]
})

# ---------------------------
# Predict button
# ---------------------------
st.markdown("---")
if st.button("🔍 Predict Bankruptcy"):
    prediction = model.predict(data)[0]

    # probabilities (robust)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(data)[0]
        classes = list(model.classes_)
        # pick indices robustly
        def find_index(targets, candidate_values):
            for t in candidate_values:
                if t in targets:
                    return targets.index(t)
            return None
        idx_bank = find_index(classes, [0, '0', 'bankruptcy', 'Bankruptcy'])
        if idx_bank is None:
            idx_bank = 0
        idx_non = 1 if idx_bank == 0 and len(classes) > 1 else (0 if len(classes)>1 else idx_bank)
        prob_bankruptcy = float(probs[idx_bank])
        prob_nonbank = float(probs[idx_non]) if idx_non < len(probs) else 1.0 - prob_bankruptcy
    else:
        prob_bankruptcy = 1.0 if prediction == 0 else 0.0
        prob_nonbank = 1.0 - prob_bankruptcy

    # result + suggestion (use dark text)
    st.markdown("---")
    if prediction == 0:
        st.error("⚠️ **Result:** The company is predicted to be at risk of **Bankruptcy**.")
        st.markdown(
            "<div class='suggestion-box' style='background:#fff2f2;border-left:6px solid #ff4d4d;padding:12px;border-radius:8px;color:#111'>"
            "💡 <b>Suggestion:</b> Improve liquidity, reduce operational risk, and enhance management efficiency."
            "</div>",
            unsafe_allow_html=True
        )
    else:
        st.success("✅ **Result:** The company is predicted to be **Financially Healthy**.")
        st.markdown(
            "<div class='suggestion-box' style='background:#f2fff4;border-left:6px solid #00aa44;padding:12px;border-radius:8px;color:#111'>"
            "💡 <b>Suggestion:</b> Maintain strong financial flexibility and competitiveness to ensure long-term stability."
            "</div>",
            unsafe_allow_html=True
        )

    # ---------------------------
    # Probabilities card (high contrast)
    # ---------------------------
    st.markdown("### 🔢 Predicted Probabilities")
    perc_bank = round(prob_bankruptcy * 100, 1)
    perc_non = round(prob_nonbank * 100, 1)

    card_html = f"""
    <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.06);border-radius:12px;padding:14px;margin-bottom:12px;box-shadow:0 4px 12px rgba(15,23,42,0.04);">
      <div style="display:flex; gap:12px; align-items:stretch;">
        <div style="flex:1;background:linear-gradient(180deg, rgba(255,107,107,0.06), rgba(255,107,107,0.02));border-radius:10px;padding:12px;min-width:140px;">
            <div style="color:#7a1f1f;font-weight:700;font-size:14px;margin-bottom:6px;">Bankruptcy</div>
            <div style="color:#111;font-weight:800;font-size:26px;">{perc_bank} %</div>
            <div style="color:#555;font-size:12px;margin-top:6px;">Probability (model)</div>
        </div>
        <div style="flex:1;background:linear-gradient(180deg, rgba(76,217,100,0.06), rgba(76,217,100,0.02));border-radius:10px;padding:12px;min-width:140px;">
            <div style="color:#11633f;font-weight:700;font-size:14px;margin-bottom:6px;">Non-Bankruptcy</div>
            <div style="color:#111;font-weight:800;font-size:26px;">{perc_non} %</div>
            <div style="color:#555;font-size:12px;margin-top:6px;">Probability (model)</div>
        </div>
      </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

    # accessible metrics
    c1, c2 = st.columns(2)
    c1.metric("Bankruptcy", f"{perc_bank} %")
    c2.metric("Non-Bankruptcy", f"{perc_non} %")

    # ---------------------------
    # Charts (matplotlib)
    # ---------------------------
    labels = ['Bankruptcy', 'Non-Bankruptcy']
    values = [prob_bankruptcy, prob_nonbank]
    colors = ['#ff6b6b', '#4cd964']

    fig1, ax1 = plt.subplots(figsize=(3.5, 3.5))
    wedges, texts, autotexts = ax1.pie(
        values,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.45, edgecolor='w')
    )
    ax1.axis('equal')
    plt.setp(autotexts, size=10, weight="bold", color="#333333")
    plt.setp(texts, size=10)
    st.pyplot(fig1, clear_figure=True)

    fig2, ax2 = plt.subplots(figsize=(6, 1.2))
    y_pos = np.arange(len(labels))
    ax2.barh(y_pos, [v * 100 for v in values], color=colors, height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlim(0, 100)
    ax2.invert_yaxis()
    ax2.set_xlabel('Probability (%)')
    for i, v in enumerate([v * 100 for v in values]):
        ax2.text(v + 1.5, i, f"{v:.1f}%", va='center', color='#111111', fontweight='600')
    plt.tight_layout()
    st.pyplot(fig2, clear_figure=True)



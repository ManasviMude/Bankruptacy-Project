import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Load trained Logistic Regression model
# ---------------------------
with open('final_logreg_model.pkl', 'rb') as f:
    model = pickle.load(f)

# ---------------------------
# Streamlit App UI
# ---------------------------
st.set_page_config(page_title="Bankruptcy Prediction App", layout="centered")
st.title("🏦 Bankruptcy Prediction App (Logistic Regression)")
st.markdown("""
Predict whether a company is likely to go **Bankrupt** or remain **Financially Healthy**  
based on its financial indicators.
""")

# Input fields
st.subheader("📊 Enter Company Financial Indicators")
industrial_risk = st.number_input("Industrial Risk", 0.0, 1.0, 0.5)
management_risk = st.number_input("Management Risk", 0.0, 1.0, 0.5)
financial_flexibility = st.number_input("Financial Flexibility", 0.0, 1.0, 0.5)
credibility = st.number_input("Credibility", 0.0, 1.0, 0.5)
competitiveness = st.number_input("Competitiveness", 0.0, 1.0, 0.5)
operating_risk = st.number_input("Operating Risk", 0.0, 1.0, 0.5)

# Prepare dataframe for prediction
data = pd.DataFrame({
    'industrial_risk': [industrial_risk],
    'management_risk': [management_risk],
    'financial_flexibility': [financial_flexibility],
    'credibility': [credibility],
    'competitiveness': [competitiveness],
    'operating_risk': [operating_risk]
})

# Prediction
if st.button("🔍 Predict Bankruptcy"):
    prediction = model.predict(data)[0]
    if prediction == 0:
        st.error("⚠️ The company is predicted to be at risk of **Bankruptcy**.")
    else:
        st.success("✅ The company is predicted to be **Financially Healthy**.")
        
# ---------------------------------------------------
# 📂 Batch Prediction via File Upload (CSV or Excel)
# ---------------------------------------------------
st.markdown("---")
st.subheader("📂 Upload File for Batch Prediction (CSV or Excel)")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file with company financial data",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file:
    # Detect file type and load appropriately
    try:
        if uploaded_file.name.endswith('.csv'):
            batch_data = pd.read_csv(uploaded_file)
        else:
            batch_data = pd.read_excel(uploaded_file)
        
        st.write("✅ File uploaded successfully!")
        st.write("Preview of uploaded data:")
        st.dataframe(batch_data.head())
        
        # Make predictions
        preds = model.predict(batch_data)
        batch_data['Prediction'] = ['Bankruptcy' if p == 0 else 'Non-Bankruptcy' for p in preds]
        
        st.markdown("### 📊 Batch Predictions:")
        st.dataframe(batch_data)
        
        # Allow user to download results
        csv_download = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv_download,
            file_name="bankruptcy_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and Scikit-learn")

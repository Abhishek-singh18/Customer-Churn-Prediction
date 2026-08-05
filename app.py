import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📊", layout="wide")

# 1. Load saved model, scaler, feature columns, and metrics
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    with open("models/metrics.json", "r") as f:
        metrics = json.load(f)
    return model, scaler, feature_columns, metrics

try:
    model, scaler, feature_columns, metrics = load_artifacts()
except Exception as e:
    st.error("Artifacts not found. Make sure you ran `python train_model.py` first.")
    st.stop()

st.title("📊 Telco Customer Churn Prediction App")

tab1, tab2 = st.tabs(["🔮 Single Customer Prediction", "📈 Model Evaluation Metrics"])

with tab1:
    st.subheader("Enter Customer Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)

    with col2:
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with col3:
        TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox(
            "Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
        TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0)

    if st.button("Predict Churn Risk", type="primary"):
        # Step A: Create DataFrame from user inputs
        raw_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }])
        
        # Step B: One-hot encode inputs
        encoded_data = pd.get_dummies(raw_data)
        
        # Step C: Align columns with training feature columns (fill missing dummy columns with 0)
        aligned_data = encoded_data.reindex(columns=feature_columns, fill_value=0)
        
        # Step D: Scale numerical features using saved scaler
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        aligned_data[num_cols] = scaler.transform(aligned_data[num_cols])
        
        # Step E: Make Prediction
        churn_prob = model.predict_proba(aligned_data)[0][1]
        churn_pred = model.predict(aligned_data)[0]
        
        st.divider()
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.metric("Churn Probability", f"{churn_prob * 100:.2f}%")
        
        with res_col2:
            if churn_pred == 1:
                st.error("⚠️ High Risk of Churn")
            else:
                st.success("✅ Low Risk of Churn")

with tab2:
    st.subheader("Model Performance Summary")
    st.json(metrics)

# --- Developer Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d; font-size: 14px; padding: 10px 0;'>"
    "Developer — <b>Abhishek Singh</b>"
    "</div>",
    unsafe_allow_html=True
)
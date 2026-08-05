# 📊 Telco Customer Churn Prediction App

A machine learning web application built with **Python**, **Scikit-Learn**, and **Streamlit** to predict customer churn risk for a telecommunications provider.

---

## 📌 Project Overview

Customer churn is a critical metric for subscription-based businesses. This project provides an end-to-end binary classification system that cleans raw customer data, pre-processes numerical and categorical features, trains a **Random Forest Classifier**, and serves real-time predictions through an interactive **Streamlit** dashboard.

---

## 🛠️ Project Structure

```text
Customer prediction churn/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Telco customer dataset
│
├── models/                                    # Exported model artifacts
│   ├── model.pkl                              # Trained Random Forest model
│   ├── scaler.pkl                             # Fitted StandardScaler
│   ├── feature_columns.pkl                    # Ordered feature vector schema
│   └── metrics.json                           # Model evaluation metrics
│
├── app.py                                     # Streamlit web application
├── train_model.py                             # Data pipeline & training script
├── requirements.txt                           # Project dependencies
└── README.md                                  # Project documentation
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

def train_and_save():
    # 1. Load Data
    print("Loading dataset...")
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # 2. Drop unique identifier
    df = df.drop(columns=["customerID"], errors="ignore")

    # 3. Clean numerical data (convert string spaces to numeric, fill missing values with median)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # 4. Map target column to binary (1 for Yes, 0 for No)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 5. One-Hot Encoding categorical features
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Separate features and target
    X = df_encoded.drop(columns=["Churn"])
    y = df_encoded["Churn"]

    # Save feature names so Streamlit can align single input rows later
    feature_columns = X.columns.tolist()

    # 6. Train-Test Split (80% training, 20% testing with stratification)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Scale numerical columns using StandardScaler
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    
    # Fit scaler on X_train and transform both sets
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # 8. Train Random Forest Model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 9. Evaluate Model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_proba), 4)
    }

    print("\n--- Model Results ---")
    for metric_name, score in metrics.items():
        print(f"{metric_name}: {score}")

    # 10. Save trained model, scaler, feature list, and metrics
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_columns, "models/feature_columns.pkl")

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nSuccess! Saved model, scaler, and feature lists inside 'models/' folder.")

if __name__ == "__main__":
    train_and_save()
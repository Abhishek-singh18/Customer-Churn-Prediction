import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



pd.set_option('display.max_columns', None)
pd.set_option('display.width', 50)


df = pd.read_csv(r'C:\Users\abhis\OneDrive\Desktop\Customer prediciton churn\WA_Fn-UseC_-Telco-Customer-Churn.csv')


df = df.drop('customerID', axis=1)


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0) 


binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})


df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})


multi_category_cols = [
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
]

df = pd.get_dummies(df, columns=multi_category_cols, drop_first=True, dtype=int)


scaler = MinMaxScaler()
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])


X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


print("Success! Text converted, data scaled, and target balanced.")
print(f"Final Training Features Shape: {X_train_resampled.shape}")
print("\nTarget distribution after SMOTE (Should be 50/50):")
print(y_train_resampled.value_counts())


print('\nTraining Random Forest Model')
rf_model = RandomForestClassifier(random_state = 42, n_estimators=100)
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred = rf_model.predict(X_test)
print('\nModel evaluation')

accuracy = accuracy_score(y_test, y_pred)
print(f'overall Accuracy: {accuracy * 100:.2f}%\n')
print('Classification reprot')
print(classification_report(y_test, y_pred))
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))


feature_importances = rf_model.feature_importances_
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})


features_df = features_df.sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 6))

sns.barplot(x='Importance', y='Feature', data=features_df.head(10), palette='viridis')
plt.title('Top 10 Most Important Features for Predicting Churn')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()


plt.show()
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

# Make sure models folder exists
os.makedirs("models", exist_ok=True)

print("="*60)
print("   MULTI-DISEASE PREDICTOR — MODEL TRAINING")
print("="*60)

# ─────────────────────────────────────────────
# 🩸 1. DIABETES MODEL
# ─────────────────────────────────────────────
print("\n[1/3] Training Diabetes Model...")

diabetes_df = pd.read_csv("datasets/diabetes.csv")

cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
diabetes_df[cols_with_zeros] = diabetes_df[cols_with_zeros].replace(0, np.nan)
diabetes_df.fillna(diabetes_df.median(numeric_only=True), inplace=True)

X_diabetes = diabetes_df.drop("Outcome", axis=1)
y_diabetes = diabetes_df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X_diabetes, y_diabetes, test_size=0.2, random_state=42
)

scaler_diabetes = StandardScaler()
X_train = scaler_diabetes.fit_transform(X_train)
X_test = scaler_diabetes.transform(X_test)

diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)
diabetes_model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, diabetes_model.predict(X_test))
print(f"    ✅ Diabetes Model Accuracy: {accuracy*100:.2f}%")

joblib.dump(diabetes_model, "models/diabetes_model.pkl")
joblib.dump(scaler_diabetes, "models/diabetes_scaler.pkl")

# ─────────────────────────────────────────────
# 🫀 2. HEART DISEASE MODEL
# ─────────────────────────────────────────────
print("\n[2/3] Training Heart Disease Model...")

heart_df = pd.read_csv("datasets/heart.csv")
heart_df.fillna(heart_df.median(numeric_only=True), inplace=True)

X_heart = heart_df.drop("target", axis=1)
y_heart = heart_df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X_heart, y_heart, test_size=0.2, random_state=42
)

scaler_heart = StandardScaler()
X_train = scaler_heart.fit_transform(X_train)
X_test = scaler_heart.transform(X_test)

heart_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
heart_model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, heart_model.predict(X_test))
print(f"    ✅ Heart Disease Model Accuracy: {accuracy*100:.2f}%")

joblib.dump(heart_model, "models/heart_model.pkl")
joblib.dump(scaler_heart, "models/heart_scaler.pkl")

# ─────────────────────────────────────────────
# 🫁 3. PARKINSON'S MODEL
# ─────────────────────────────────────────────
print("\n[3/3] Training Parkinson's Model...")

parkinsons_df = pd.read_csv("datasets/parkinsons.csv")

# Drop name column FIRST before any numeric operations
if 'name' in parkinsons_df.columns:
    parkinsons_df.drop("name", axis=1, inplace=True)

parkinsons_df.fillna(parkinsons_df.median(numeric_only=True), inplace=True)

X_parkinsons = parkinsons_df.drop("status", axis=1)
y_parkinsons = parkinsons_df["status"]

X_train, X_test, y_train, y_test = train_test_split(
    X_parkinsons, y_parkinsons, test_size=0.2, random_state=42
)

scaler_parkinsons = StandardScaler()
X_train = scaler_parkinsons.fit_transform(X_train)
X_test = scaler_parkinsons.transform(X_test)

parkinsons_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
parkinsons_model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, parkinsons_model.predict(X_test))
print(f"    ✅ Parkinson's Model Accuracy: {accuracy*100:.2f}%")

joblib.dump(parkinsons_model, "models/parkinsons_model.pkl")
joblib.dump(scaler_parkinsons, "models/parkinsons_scaler.pkl")

# ─────────────────────────────────────────────
print("\n" + "="*60)
print("   ALL 3 MODELS TRAINED & SAVED SUCCESSFULLY! 🎉")
print("="*60)
print("\nFiles saved in models/:")
print("  - diabetes_model.pkl + diabetes_scaler.pkl")
print("  - heart_model.pkl    + heart_scaler.pkl")
print("  - parkinsons_model.pkl + parkinsons_scaler.pkl")
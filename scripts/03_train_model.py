import joblib
import os
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data
p = '../data/processed'
X_train, y_train = joblib.load(f'{p}/X_train.joblib'), joblib.load(f'{p}/y_train.joblib')
X_val, y_val = joblib.load(f'{p}/X_val.joblib'), joblib.load(f'{p}/y_val.joblib')


# XGBoost
print("🚀 Training XGBoost...")
xgb = XGBClassifier(n_estimators=300, learning_rate=0.098, max_depth=4, random_state=42, scale_pos_weight=1.12)
xgb.fit(X_train, y_train)

preds = xgb.predict(X_val)
print(f"\n--- XGBoost Results ---")
print(f"Accuracy: {accuracy_score(y_val, preds):.4f}")
print(classification_report(y_val, preds))
    
# Define path to save models
model_save_path_xgb = '../xgboost.joblib'


# 1. Save the XGBoost model
joblib.dump(xgb, model_save_path_xgb)
print(f"✅ XGBoost model saved to: {model_save_path_xgb}")

# 3. Double check the file exists
if os.path.exists(model_save_path_xgb):
    print("📁 Verification: Model file is physically on disk.")
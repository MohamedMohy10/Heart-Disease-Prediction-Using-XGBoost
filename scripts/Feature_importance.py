import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Load the model and the original column names
p = '../data/processed'
model = joblib.load('../xgboost_89.joblib')

# --- Get the Feature Names ---
raw_df = pd.read_csv('../data/heart_60k.csv').drop(['id', 'Heart Disease'], axis=1, errors='ignore')
cat_cols = ['Chest pain type', 'EKG results', 'Slope of ST', 'Thallium']
df_encoded = pd.get_dummies(raw_df, columns=cat_cols, drop_first=True)
feature_names = df_encoded.columns.tolist()

# 2. Get Feature Importance
importances = model.feature_importances_
# Create the DataFrame with actual names
feature_data = pd.DataFrame({
    'Feature': feature_names, 
    'Importance': importances
})

# Sort by importance
feature_data = feature_data.sort_values(by='Importance', ascending=False)

# 3. Plotting
plt.figure(figsize=(12, 8))
sns.barplot(
    x='Importance', 
    y='Feature', 
    data=feature_data.head(10), 
    palette='magma'
)

plt.title('Top 10 Most Important Features for Heart Disease (XGBoost)')
plt.xlabel('Importance Score (Gain)')
plt.ylabel('Feature Name')
plt.tight_layout() # Ensures labels don't get cut off
plt.savefig('feature_importance_named.png')

print("✅ Success! Chart with feature names saved as 'feature_importance.png'.")
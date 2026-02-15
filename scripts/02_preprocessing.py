import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

file_path = '../data/heart_60k.csv'
df = pd.read_csv(file_path)

# 1. Cleaning
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# 2. Encoding
le = LabelEncoder()
df['Heart Disease'] = le.fit_transform(df['Heart Disease'])

# One-hot encoding categorical variables
cat_cols = ['Chest pain type', 'EKG results', 'Slope of ST', 'Thallium']
df_final = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 3. Split (70/15/15)
X = df_final.drop('Heart Disease', axis=1)
y = df_final['Heart Disease']

# First 70% Train, 30% Temp
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
# Split Temp into 15% Val and 15% Test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# 4. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 5. Save
processed_dir = '../data/processed'
os.makedirs(processed_dir, exist_ok=True)

joblib.dump(X_train_scaled, f'{processed_dir}/X_train.joblib')
joblib.dump(X_val_scaled, f'{processed_dir}/X_val.joblib')
joblib.dump(X_test_scaled, f'{processed_dir}/X_test.joblib')
joblib.dump(y_train, f'{processed_dir}/y_train.joblib')
joblib.dump(y_val, f'{processed_dir}/y_val.joblib')
joblib.dump(y_test, f'{processed_dir}/y_test.joblib')
joblib.dump(scaler, f'{processed_dir}/scaler.joblib')

print(f"✅ Preprocessing complete. Training on {len(X_train)} samples.")
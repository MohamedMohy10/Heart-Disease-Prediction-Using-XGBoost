import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'heart_60k.csv')
df = pd.read_csv(file_path)

print("--- Data Overview ---")
print(df.info())

# 1. Drop ID if it exists
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# 2. Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum().sum())

# 3. Target Distribution
print("\n--- Target Balance ---")
print(df['Heart Disease'].value_counts(normalize=True) * 100)

# 4. Outlier Check for Numerical Data
plt.figure(figsize=(12, 6))
df[['BP', 'Cholesterol', 'Max HR']].boxplot()
plt.title("Checking for Outliers in BP, Cholesterol, and Max HR")
plt.savefig('large_outliers.png')
print("\n📊 Outlier boxplot saved as 'large_outliers.png'.")
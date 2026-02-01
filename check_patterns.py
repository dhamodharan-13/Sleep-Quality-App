import pandas as pd
import numpy as np

# Load the dataset
try:
    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    print("Dataset Loaded Successfully.")
except FileNotFoundError:
    print("Error: Dataset not found.")
    exit()

# Clean data (same as training script)
df = df.dropna()

print("\n--- 1. ANALYSIS: Sleep Duration vs Quality ---")
# Group by rounded sleep duration to see the pattern
df['Sleep_Round'] = df['Sleep Duration'].round()
grouped = df.groupby('Sleep_Round')['Quality of Sleep'].agg(['mean', 'count', 'min', 'max'])
print(grouped)

print("\n--- 2. Correlation Matrix ---")
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()['Quality of Sleep'].sort_values(ascending=False)
print(corr)

print("\n--- 3. 'Exceptions' Check ---")
# Are there people with HIGH sleep but LOW quality?
weird_cases = df[(df['Sleep Duration'] >= 8) & (df['Quality of Sleep'] < 7)]
print(f"\nPeople with 8+ hours sleep but < 7 score: {len(weird_cases)}")

if len(weird_cases) > 0:
    print(weird_cases[['Age', 'Sleep Duration', 'Quality of Sleep', 'Stress Level', 'BMI Category']].head())
else:
    print("No exceptions found. High sleep always equals high score in this dataset.")

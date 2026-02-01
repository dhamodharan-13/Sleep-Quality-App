"""
Sleep Quality Prediction Model - Enhanced Training
Includes: Cross-Validation, Multiple Models, Comprehensive Metrics, Input Validation Rules
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import warnings
import warnings
import seaborn as sns # Ensure seaborn is imported
warnings.filterwarnings('ignore')

# --- 1. Load & Clean Data ---
print("=" * 60)
print("SLEEP QUALITY PREDICTION - MODEL TRAINING")
print("=" * 60)

print("\n[1/6] Loading dataset...")
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Remove empty/malformed rows
initial_rows = len(df)
df = df.dropna()
df = df[df['Person ID'].apply(lambda x: str(x).isdigit())]
cleaned_rows = len(df)
print(f"   Dataset: {initial_rows} rows -> {cleaned_rows} after cleaning")

# --- 2. Feature Engineering ---
print("\n[2/6] Feature Engineering...")

# Drop Person ID (not a useful feature)
df = df.drop(columns=["Person ID"])

# Drop columns not used in user input (as requested)
# We are removing 'Blood Pressure' and 'Heart Rate' only
# Sleep Duration is NOW INCLUDED for better prediction responsiveness
df = df.drop(columns=["Blood Pressure", "Heart Rate"])

# --- 3. Encoding Categorical Variables ---
categorical_cols = ["Gender", "Occupation", "BMI Category", "Sleep Disorder"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print(f"   Features after encoding: {df.shape[1] - 1}")

# --- 4. Split Features and Target ---
X = df.drop(columns=["Quality of Sleep"])
y = df["Quality of Sleep"]

# Store feature names for later use
feature_names = X.columns.tolist()

# --- 5. Train-Test Split ---
print("\n[3/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   Training: {len(X_train)} samples | Test: {len(X_test)} samples")

# --- 6. Define Valid Input Ranges (for API validation) ---
valid_ranges = {
    "Age": {"min": 10, "max": 80, "default": 30},
    "Sleep Duration": {"min": 1, "max": 12, "default": 7},
    "Stress Level": {"min": 1, "max": 10, "default": 5},
    "Physical Activity Level": {"min": 0, "max": 300, "default": 45},
    "Daily Steps": {"min": 100, "max": 30000, "default": 6000}
}

# --- 7. Cross-Validation with Multiple Models ---
print("\n[4/6] Cross-Validation (5-Fold)...")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

cv_results = {}
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
    cv_results[name] = {
        "mean_r2": scores.mean(),
        "std_r2": scores.std(),
        "scores": scores
    }
    print(f"\n   {name}:")
    print(f"      Fold Scores: {[f'{s:.3f}' for s in scores]}")
    print(f"      Mean R²: {scores.mean():.4f} (±{scores.std():.4f})")

# Select best model
best_model_name = max(cv_results, key=lambda x: cv_results[x]["mean_r2"])
print(f"\n   [OK] Best Model: {best_model_name}")

# --- 8. Train Final Model ---
print("\n[5/6] Training final model...")
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# --- 9. Evaluate on Test Set ---
y_pred = best_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("TEST SET EVALUATION RESULTS")
print("=" * 60)
print(f"   Mean Squared Error (MSE):  {mse:.4f}")
print(f"   Root MSE (RMSE):           {rmse:.4f}")
print(f"   Mean Absolute Error (MAE): {mae:.4f}")
print(f"   R² Score:                  {r2:.4f}")
print("=" * 60)

# --- 10. Feature Importance ---
print("\n[6/6] Feature Importance Analysis...")
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": best_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n   Top 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    bar = "#" * int(row["Importance"] * 50)
    print(f"   {row['Feature']:30} {row['Importance']:.3f} {bar}")

# --- 11a. Print Linear Regression Formula (if available) ---
if "Linear Regression" in models:
    lr_model = models["Linear Regression"]
    lr_model.fit(X_train, y_train) # Ensure it's fitted
    print("\n" + "=" * 60)
    print(" EXPLICIT MODEL FORMULA (Linear Regression)")
    print("=" * 60)
    print(f"Base Quality = {lr_model.intercept_:.2f}")
    features_desc = []
    for name, coef in zip(feature_names, lr_model.coef_):
        sign = "+" if coef >= 0 else ""
        print(f"   {sign} ({coef:.2f} * {name})")
        features_desc.append(f"{sign} {coef:.2f}*{name}")
    
    print("\n   Full Formula:")
    print(f"   Quality = {lr_model.intercept_:.2f} {' '.join(features_desc)}")
    print("=" * 60)

# --- 11. Save Feature Importance Plot ---
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
colors = plt.cm.RdPu(np.linspace(0.3, 0.9, 10))
plt.barh(top_features["Feature"], top_features["Importance"], color=colors)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importances (Modified Feature Set)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
print("\n   [OK] Saved: feature_importance.png")

# --- 11b. Save Correlation Matrix (Visualizing Patterns) ---
import seaborn as sns
plt.figure(figsize=(10, 8))
# Calculate correlation only for numeric columns
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
# Plot heatmap
sns.heatmap(correlation_matrix[['Quality of Sleep']].sort_values(by='Quality of Sleep', ascending=False),
            annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation with Sleep Quality")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches='tight')
print("   [OK] Saved: correlation_heatmap.png")

# --- 12. Save Model & Metadata ---
print("\nSaving model and metadata...")
model_data = {
    "model": best_model,
    "features": feature_names,
    "valid_ranges": valid_ranges,
    "metrics": {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "cv_mean_r2": cv_results[best_model_name]["mean_r2"],
        "cv_std_r2": cv_results[best_model_name]["std_r2"]
    },
    "model_name": best_model_name
}
joblib.dump(model_data, "sleep_model.pkl")

print("\n" + "=" * 60)
print("[OK] MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"   Model: {best_model_name}")
print(f"   Test R²: {r2:.4f}")
print(f"   Cross-Val R²: {cv_results[best_model_name]['mean_r2']:.4f}")
print(f"   Saved to: sleep_model.pkl")
print("=" * 60)

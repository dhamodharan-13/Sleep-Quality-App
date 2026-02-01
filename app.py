"""
Sleep Quality Prediction API
Flask backend with robust input validation and meaningful error messages
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import os
import shap

app = Flask(__name__)
CORS(app)

# Serve the HTML frontend
@app.route('/')
def home():
    return send_file('index.html')

# Load Model & Metadata
print("Loading model...")
model_data = joblib.load("sleep_model.pkl")
model = model_data["model"]
model_columns = model_data["features"]
valid_ranges = model_data.get("valid_ranges", {})
model_metrics = model_data.get("metrics", {})
model_name = model_data.get("model_name", "Unknown")

print(f"✓ Model loaded: {model_name}")
print(f"  Test R²: {model_metrics.get('r2', 'N/A'):.4f}" if model_metrics.get('r2') else "  Metrics not available")

# Default valid ranges (fallback if not in model)
DEFAULT_RANGES = {
    "Age": {"min": 10, "max": 80, "default": 30},
    "Sleep Duration": {"min": 1, "max": 12, "default": 7},
    "Stress Level": {"min": 1, "max": 10, "default": 5},
    "Physical Activity Level": {"min": 0, "max": 300, "default": 45},
    "Daily Steps": {"min": 100, "max": 30000, "default": 6000}
}

# Use model ranges or defaults
VALID_RANGES = valid_ranges if valid_ranges else DEFAULT_RANGES


def validate_input(data):
    """
    Validate input data and return errors list.
    Returns (cleaned_data, errors)
    """
    errors = []
    cleaned = {}
    
    # Required numeric fields to validate (including Sleep Duration now)
    numeric_fields = [
        ("Age", "Age"),
        ("Sleep Duration", "Sleep Duration"),
        ("Stress Level", "Stress Level"),
        ("Daily Steps", "Daily Steps"),
        ("Physical Activity Level", "Physical Activity Level")
    ]
    
    for field, display_name in numeric_fields:
        if field in data:
            try:
                value = float(data[field])
                
                # Check range if defined
                if field in VALID_RANGES:
                    min_val = VALID_RANGES[field]["min"]
                    max_val = VALID_RANGES[field]["max"]
                    
                    if value < min_val or value > max_val:
                        errors.append(f"{display_name} must be between {min_val} and {max_val} (got: {value})")
                    else:
                        cleaned[field] = value
                else:
                    cleaned[field] = value
                    
            except (ValueError, TypeError):
                errors.append(f"{display_name} must be a valid number")
        else:
            # Use default if available
            if field in VALID_RANGES:
                cleaned[field] = VALID_RANGES[field]["default"]
    
    # Validate categorical fields
    valid_genders = ["Male", "Female", "Other"]
    if "Gender" in data:
        if data["Gender"] not in valid_genders:
            errors.append(f"Gender must be one of: {', '.join(valid_genders)}")
        else:
            cleaned["Gender"] = data["Gender"]
    
    valid_bmi = ["Normal", "Overweight", "Obese", "Normal Weight"]
    if "BMI Category" in data:
        if data["BMI Category"] not in valid_bmi:
            errors.append(f"BMI Category must be one of: {', '.join(valid_bmi)}")
        else:
            cleaned["BMI Category"] = data["BMI Category"]
    
    valid_disorders = ["None", "Insomnia", "Sleep Apnea"]
    if "Sleep Disorder" in data:
        if data["Sleep Disorder"] not in valid_disorders:
            errors.append(f"Sleep Disorder must be one of: {', '.join(valid_disorders)}")
        else:
            cleaned["Sleep Disorder"] = data["Sleep Disorder"]
    
    # Copy other fields
    if "Occupation" in data:
        cleaned["Occupation"] = data["Occupation"]
    
    # Age-Occupation Logical Validation
    age = cleaned.get("Age", 30)
    occupation = cleaned.get("Occupation", "")
    
    # Define minimum ages for professional occupations
    occupation_min_ages = {
        "Doctor": 24,
        "Lawyer": 23,
        "Nurse": 21,
        "Engineer": 22,
        "Scientist": 24,
        "Accountant": 22,
        "Software Engineer": 21
    }
    
    # Define maximum ages for certain occupations
    occupation_max_ages = {
        "Student": 35  # Reasonable max for students
    }
    
    if occupation in occupation_min_ages:
        min_age = occupation_min_ages[occupation]
        if age < min_age:
            errors.append(f"A {occupation} must be at least {min_age} years old (you entered {int(age)})")
    
    if occupation in occupation_max_ages:
        max_age = occupation_max_ages[occupation]
        if age > max_age:
            errors.append(f"Age {int(age)} seems too high for a {occupation}. Max expected: {max_age}")
    
    return cleaned, errors


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)
        
        # Validate input
        cleaned_data, validation_errors = validate_input(data)
        
        if validation_errors:
            return jsonify({
                "status": "error",
                "error": "Validation failed",
                "validation_errors": validation_errors,
                "message": "Please correct the following inputs:\n" + "\n".join(f"• {e}" for e in validation_errors)
            }), 400
        
        # Create DataFrame with all model columns initialized to 0
        final_input = pd.DataFrame(columns=model_columns)
        final_input.loc[0] = 0
        
        # Fill numeric values (including Sleep Duration now)
        numeric_cols = ["Age", "Sleep Duration", "Physical Activity Level", "Stress Level", "Daily Steps"]
        
        for col in numeric_cols:
            if col in cleaned_data:
                final_input.at[0, col] = float(cleaned_data[col])
            elif col in VALID_RANGES:
                final_input.at[0, col] = VALID_RANGES[col]["default"]
        
        # Handle categorical encoding (One-Hot)
        if "Gender" in cleaned_data:
            col_name = f"Gender_{cleaned_data['Gender']}"
            if col_name in model_columns:
                final_input.at[0, col_name] = 1
                
        if "Occupation" in cleaned_data:
            col_name = f"Occupation_{cleaned_data['Occupation']}"
            if col_name in model_columns:
                final_input.at[0, col_name] = 1
                
        if "BMI Category" in cleaned_data:
            col_name = f"BMI Category_{cleaned_data['BMI Category']}"
            if col_name in model_columns:
                final_input.at[0, col_name] = 1
                
        if "Sleep Disorder" in cleaned_data:
            col_name = f"Sleep Disorder_{cleaned_data['Sleep Disorder']}"
            if col_name in model_columns:
                final_input.at[0, col_name] = 1

        # Make prediction
        prediction = model.predict(final_input)[0]
        prediction = float(np.clip(prediction, 1, 10))  # Ensure prediction is within valid range
        prediction = round(prediction, 1)

        # Generate recommendations based on input
        recommendations = []
        
        stress_level = float(cleaned_data.get("Stress Level", 5))
        daily_steps = float(cleaned_data.get("Daily Steps", 6000))
        activity_level = float(cleaned_data.get("Physical Activity Level", 45))
        
        if stress_level > 6:
            recommendations.append("🧘 High stress detected. Try meditation or relaxation techniques.")
        if daily_steps < 5000:
            recommendations.append("🚶 Aim for at least 5,000 steps daily for better health.")
        if activity_level < 30:
            recommendations.append("🏃 Low activity level. Consider adding more exercise to your routine.")
        if prediction < 6:
            recommendations.append("📊 Your sleep quality is below average. Review your lifestyle habits.")
        
        if not recommendations:
            recommendations.append("✅ Great job! Keep maintaining your healthy lifestyle.")

        # SHAP-based dynamic feature contribution (changes based on input values)
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(final_input)
            
            # Calculate absolute contribution percentages
            shap_abs = np.abs(shap_values[0])
            total_shap = np.sum(shap_abs)
            
            if total_shap > 0:
                shap_percentages = (shap_abs / total_shap) * 100
            else:
                shap_percentages = np.zeros_like(shap_abs)
            
            # Build feature list with SHAP contributions
            active_features = []
            for i, col in enumerate(model_columns):
                if shap_percentages[i] > 0.5:  # Only show features with > 0.5% contribution
                    active_features.append({
                        "name": col.replace("_", " ").replace("Occupation ", "").replace("BMI Category ", "").replace("Sleep Disorder ", ""),
                        "importance": round(shap_percentages[i], 1),
                        "value": float(final_input.at[0, col]),
                        "contribution": round(float(shap_values[0][i]), 2)  # Positive = helps, Negative = hurts
                    })
            
            # Identify critical factors
            for feat in active_features:
                feat["is_critical"] = False
                name_lower = feat["name"].lower()
                val = feat["value"]
                
                # Logic for marking severity
                if "stress level" in name_lower and val >= 8:
                    feat["is_critical"] = True
                elif "sleep duration" in name_lower and val < 6:
                    feat["is_critical"] = True
                elif "daily steps" in name_lower and val < 4000:
                    feat["is_critical"] = True
                elif "sleep disorder" in name_lower and "insomnia" in name_lower and val == 1:
                    feat["is_critical"] = True
                elif "sleep disorder" in name_lower and "apnea" in name_lower and val == 1:
                    feat["is_critical"] = True

            # Sort by Criticality first, then Importance
            active_features.sort(key=lambda x: (x["is_critical"], x["importance"]), reverse=True)
            top_features = active_features[:5]
        except Exception as e:
            # Fallback to static feature importance if SHAP fails
            print(f"SHAP calculation failed: {e}")
            feature_importances = model.feature_importances_
            importance_dict = dict(zip(model_columns, feature_importances))
            
            active_features = []
            for col in model_columns:
                if final_input.at[0, col] != 0:
                    active_features.append({
                        "name": col.replace("_", " ").replace("Occupation ", "").replace("BMI Category ", "").replace("Sleep Disorder ", ""),
                        "importance": round(importance_dict[col] * 100, 1),
                        "value": float(final_input.at[0, col])
                    })
            
            active_features.sort(key=lambda x: x["importance"], reverse=True)
            top_features = active_features[:5]

        # What-If Analysis (updated for new features)
        what_if = []
        
        if stress_level > 3:
            test_input = final_input.copy()
            test_input.at[0, "Stress Level"] = max(1, stress_level - 2)
            new_pred = model.predict(test_input)[0]
            if new_pred > prediction:
                what_if.append({
                    "scenario": "-2 stress level",
                    "improvement": round(new_pred - prediction, 1)
                })
        
        if daily_steps < 10000:
            test_input = final_input.copy()
            test_input.at[0, "Daily Steps"] = daily_steps + 2000
            new_pred = model.predict(test_input)[0]
            if new_pred > prediction:
                what_if.append({
                    "scenario": "+2000 daily steps",
                    "improvement": round(new_pred - prediction, 1)
                })

        response = {
            "prediction": prediction,
            "recommendations": recommendations,
            "feature_importance": top_features,
            "what_if": what_if,
            "model_info": {
                "name": model_name,
                "r2_score": round(model_metrics.get("r2", 0), 4)
            },
            "status": "success"
        }
        
        return jsonify(response)

    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e), 
            "status": "error",
            "message": "An unexpected error occurred. Please try again."
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": model_name,
        "features_count": len(model_columns)
    })


@app.route('/valid-ranges', methods=['GET'])
def get_valid_ranges():
    """Return valid input ranges for frontend validation"""
    return jsonify(VALID_RANGES)


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("SLEEP QUALITY PREDICTION API")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Endpoints:")
    print(f"  POST /predict     - Make predictions")
    print(f"  GET  /health      - Health check")
    print(f"  GET  /valid-ranges - Get input validation ranges")
    print("=" * 50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)

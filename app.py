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
    "Age": {"min": 10, "max": 100, "default": 30},
    "Sleep Duration": {"min": 1, "max": 14, "default": 7},
    "Stress Level": {"min": 1, "max": 10, "default": 5},
    "Physical Activity Level": {"min": 0, "max": 300, "default": 45},
    "Heart Rate": {"min": 40, "max": 150, "default": 70},
    "Daily Steps": {"min": 100, "max": 30000, "default": 6000},
    "Systolic_BP": {"min": 70, "max": 200, "default": 120},
    "Diastolic_BP": {"min": 40, "max": 130, "default": 80}
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
    
    # Required numeric fields to validate
    numeric_fields = [
        ("Age", "Age"),
        ("Sleep Duration", "Sleep Duration"),
        ("Stress Level", "Stress Level"),
        ("Heart Rate", "Heart Rate"),
        ("Daily Steps", "Daily Steps"),
        ("Systolic_BP", "Systolic Blood Pressure"),
        ("Diastolic_BP", "Diastolic Blood Pressure"),
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
    valid_genders = ["Male", "Female"]
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
        
        # Fill numeric values
        numeric_cols = ["Age", "Sleep Duration", "Physical Activity Level", "Stress Level", 
                        "Heart Rate", "Daily Steps", "Systolic_BP", "Diastolic_BP"]
        
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
        
        sleep_duration = float(cleaned_data.get("Sleep Duration", 7))
        stress_level = float(cleaned_data.get("Stress Level", 5))
        heart_rate = float(cleaned_data.get("Heart Rate", 70))
        daily_steps = float(cleaned_data.get("Daily Steps", 6000))
        
        if sleep_duration < 7:
            recommendations.append("💤 Increase sleep to at least 7 hours for optimal rest.")
        if stress_level > 6:
            recommendations.append("🧘 High stress detected. Try meditation or relaxation techniques.")
        if heart_rate > 80:
            recommendations.append("❤️ Elevated heart rate. Consider cardiovascular exercises.")
        if daily_steps < 5000:
            recommendations.append("🚶 Aim for at least 5,000 steps daily for better health.")
        if prediction < 6:
            recommendations.append("📊 Your sleep quality is below average. Review your lifestyle habits.")
        
        if not recommendations:
            recommendations.append("✅ Great job! Keep maintaining your healthy lifestyle.")

        # Feature importance for this prediction
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

        # What-If Analysis
        what_if = []
        
        if sleep_duration < 9:
            test_input = final_input.copy()
            test_input.at[0, "Sleep Duration"] = sleep_duration + 1
            new_pred = model.predict(test_input)[0]
            if new_pred > prediction:
                what_if.append({
                    "scenario": "+1 hour sleep",
                    "improvement": round(new_pred - prediction, 1)
                })
        
        if stress_level > 3:
            test_input = final_input.copy()
            test_input.at[0, "Stress Level"] = max(1, stress_level - 2)
            new_pred = model.predict(test_input)[0]
            if new_pred > prediction:
                what_if.append({
                    "scenario": "-2 stress level",
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

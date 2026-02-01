"""
Sleep Quality Prediction API
Flask backend with robust input validation and meaningful error messages
"""

from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import pandas as pd
import joblib
import numpy as np
import os
import shap

app = Flask(__name__)
app.secret_key = 'super-secret-key-change-this-in-production'  # Required for session
# Database Config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    # Profile Info (Optional, updated on first prediction)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(50))
    occupation = db.Column(db.String(100))
    logs = db.relationship('SleepLog', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class SleepLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.Date, nullable=False) # One entry per day
    
    # Input Data
    sleep_duration = db.Column(db.Float)
    stress_level = db.Column(db.Integer)
    daily_steps = db.Column(db.Integer)
    
    # Result Data
    quality_score = db.Column(db.Float)
    sleep_disorder = db.Column(db.String(100))
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
CORS(app)

# Create DB
with app.app_context():
    db.create_all()

# Routes
@app.route('/')
@login_required
def home():
    return render_template('index.html', user=current_user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')
            
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username')
    password = request.form.get('password')
    
    if User.query.filter_by(username=username).first():
        flash('Username already exists', 'error')
    else:
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created! Please log in.', 'success')
        
    return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/history')
@login_required
def history():
    logs = SleepLog.query.filter_by(user_id=current_user.id).order_by(SleepLog.date.asc()).limit(30).all()
    history_data = [{
        "date": log.date.strftime('%Y-%m-%d'),
        "quality_score": log.quality_score,
        "sleep_duration": log.sleep_duration
    } for log in logs]
    return jsonify(history_data)

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

# Define Categorical Features for OHE
categorical_features = {
    "Gender": ["Male"], 
    "Occupation": [
        "Doctor", "Engineer", "Lawyer", "Nurse", 
        "Sales Representative", "Salesperson", 
        "Scientist", "Software Engineer", "Teacher"
    ],
    "BMI Category": ["Normal Weight", "Obese", "Overweight"],
    "Sleep Disorder": ["Sleep Apnea"] 
}


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
        ("Physical Activity Level", "Physical Activity Level"),
        ("Heart Rate", "Heart Rate")
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
            # Fallback default for Heart Rate if not in VALID_RANGES
            elif field == "Heart Rate":
                 cleaned[field] = 70

    # Blood Pressure (Pass through splitting if needed, or raw)
    # Assuming model handles or feature engineering happens in predict?
    # Actually, let's just pass it through.
    if "Blood Pressure" in data:
        cleaned["Blood Pressure"] = data["Blood Pressure"]
    else:
        cleaned["Blood Pressure"] = "120/80" # Default
    
    # Validate categorical fields
    valid_genders = ["Male", "Female", "Other"]
    if "Gender" in data:
        if data["Gender"] not in valid_genders:
            errors.append(f"Gender must be one of: {', '.join(valid_genders)}")
        else:
            cleaned["Gender"] = data["Gender"]
    
    valid_bmi = ["Normal", "Overweight", "Obese", "Normal Weight"]
    if "BMI Category" in data:
        val = data["BMI Category"]
        if val == "Normal": # Map Normal to Normal Weight per model columns
            val = "Normal Weight"
            
        if val not in valid_bmi:
            errors.append(f"BMI Category must be one of: {', '.join(valid_bmi)}")
        else:
            cleaned["BMI Category"] = val
    
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
@login_required
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
        
        # Prepare input for model
        input_data = pd.DataFrame([cleaned_data])
        
        # Initialize all model features with 0
        for feature in model_columns:
            if feature not in input_data.columns:
                input_data[feature] = 0
        
        
        # One-hot encode categorical features
        for col, categories in categorical_features.items():
            if col in cleaned_data: # Check in cleaned_data (raw values)
                val = cleaned_data[col]
                for cat in categories:
                    col_name = f"{col}_{cat}"
                    if col_name in input_data.columns: 
                         input_data[col_name] = 1 if val == cat else 0

        # Final filtering to ensure correct order
        input_data = input_data[model_columns]


        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 1)

        # Generate recommendations
        recommendations = []
        
        # Recommendations logic (simplified for brevity, assuming existing logic follows)
        if prediction < 6:
            recommendations.append("⚠️ Your sleep quality is low. Try to maintain a consistent sleep schedule.")
        elif prediction < 8:
             recommendations.append("ℹ️ Good sleep, but there's room for improvement.")
        else:
            recommendations.append("✅ Great job! Keep maintaining your healthy lifestyle.")

        # Alias input_data to final_input for SHAP/Fallback logic using it
        final_input = input_data

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

        # Save to History (DB)
        try:
            if current_user.is_authenticated:
                # Check if entry exists for today to update, or just create new?
                # For simplicity, we create new. Or maybe update latest?
                # Comment says "One entry per day"
                today = datetime.utcnow().date()
                existing_log = SleepLog.query.filter_by(user_id=current_user.id, date=today).first()
                
                if existing_log:
                    # Update existing
                    existing_log.sleep_duration = cleaned_data.get("Sleep Duration")
                    existing_log.stress_level = cleaned_data.get("Stress Level")
                    existing_log.daily_steps = cleaned_data.get("Daily Steps")
                    existing_log.quality_score = prediction
                    existing_log.sleep_disorder = cleaned_data.get("Sleep Disorder", "None")
                    existing_log.created_at = datetime.utcnow()
                else:
                    # Create new
                    new_log = SleepLog(
                        user_id=current_user.id,
                        date=today,
                        sleep_duration=cleaned_data.get("Sleep Duration"),
                        stress_level=cleaned_data.get("Stress Level"),
                        daily_steps=cleaned_data.get("Daily Steps"),
                        quality_score=prediction,
                        sleep_disorder=cleaned_data.get("Sleep Disorder", "None")
                    )
                    db.session.add(new_log)
                
                db.session.commit()
        except Exception as db_err:
            print(f"Database error: {db_err}")
            # Don't fail the request just because history save failed

        # What-If Analysis (updated for new features)
        what_if = []
        
        stress_level = cleaned_data.get("Stress Level", 5)
        daily_steps = cleaned_data.get("Daily Steps", 6000)
        
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
        with open("debug_error.log", "w") as f:
            f.write(str(e) + "\n")
            traceback.print_exc(file=f)
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

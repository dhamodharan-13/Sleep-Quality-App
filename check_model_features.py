import joblib
import pandas as pd

try:
    print("Loading model...")
    model_data = joblib.load("sleep_model.pkl")
    print("\n--- PICKLE KEYS ---")
    print(model_data.keys())
    
    model = model_data["model"]
    print(f"\nModel Type: {type(model)}")
    
    if "scaler" in model_data:
        print("Scaler found in pickle.")
    else:
        print("Scaler NOT found in pickle.")
        
except Exception as e:
    print(f"Error loading model: {e}")

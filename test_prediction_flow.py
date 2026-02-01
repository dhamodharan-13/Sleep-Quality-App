import requests
import json

BASE_URL = "http://localhost:5000"
SESSION = requests.Session()

def test_flow():
    print("1. Registering new user...")
    reg_payload = {"username": "autotest_user", "password": "password123"}
    # Note: The app uses a form for registration, not JSON, based on previous login.html analysis
    # Let's check app.py... it uses request.form
    resp = SESSION.post(f"{BASE_URL}/register", data=reg_payload)
    print(f"   Registration Response: {resp.status_code}")
    
    print("2. Logging in...")
    login_payload = {"username": "autotest_user", "password": "password123"}
    resp = SESSION.post(f"{BASE_URL}/login", data=login_payload)
    print(f"   Login Response: {resp.status_code}")
    
    # Verify login by checking session cookie or accessing protected route
    if "session" not in SESSION.cookies:
        print("   WARNING: No session cookie found. Login might have failed.")
    
    print("3. Testing Prediction Endpoint (The Analysis)...")
    predict_payload = {
        "Age": 30,
        "Gender": "Male",
        "Occupation": "Software Engineer",
        "Sleep Duration": 7.5,
        "Quality of Sleep": 0,
        "Physical Activity Level": 45,  # This was the missing field
        "Stress Level": 5,
        "BMI Category": "Normal",
        "Blood Pressure": "120/80",
        "Heart Rate": 70,
        "Daily Steps": 8000,
        "Sleep Disorder": "None"
    }
    
    headers = {"Content-Type": "application/json"}
    resp = SESSION.post(f"{BASE_URL}/predict", json=predict_payload, headers=headers)
    print(f"   Prediction Response Code: {resp.status_code}")
    
    if resp.status_code == 200:
        try:
            data = resp.json()
            if data.get("status") == "success":
                print(f"   SUCCESS! Prediction received: {data.get('prediction')}")
                print(f"   Comparisons: {data.get('comparison')}")
            else:
                print(f"   FAILED: API returned error: {data}")
        except json.JSONDecodeError:
             print(f"   FAILED: Non-JSON response (likely HTML error page). Response text preview: {resp.text[:200]}")
    else:
        print(f"   FAILED: Server error. Response text preview: {resp.text[:200]}")

if __name__ == "__main__":
    try:
        test_flow()
    except Exception as e:
        print(f"An error occurred: {e}")

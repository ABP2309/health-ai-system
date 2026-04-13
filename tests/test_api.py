
import requests
import json
import numpy as np

BASE_URL = "http://localhost:5000"

def test_health():
    print("--- Testing Health Endpoint ---")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

def test_predict_normal():
    print("\n--- Testing Predict: Normal Scenario ---")
    # fall_rf expects 9 features (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, acc2_x, acc2_y, acc2_z)
    payload = {
        "vitals": {
            "age": 70, "hr": 72, "sbp": 120, "dbp": 80, "map": 93, 
            "temp": 36.6, "spo2": 98, "resp": 16, "wbc": 8, "lactate": 1.1,
            "glucose": 100, "sofa": 0, "qsofa": 0
        },
        "vitals_series": np.zeros((100, 3)).tolist(),
        "behavioral": { "acc": [0.1, 0.1, 9.8, 0.0, 0.0, 0.0, 0.1, 0.1, 9.8] },
        "behavioral_stats": [0.1, 0.1, 9.8, 0.05, 0.05, 0.05, 9.8],
        "environment": { "camera_motion": True }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Alert Level: {response.json().get('alert_level')}")
        print(f"Predictions: {response.json().get('predictions')}")
    except Exception as e:
        print(f"Error: {e}")

def test_predict_critical_fall():
    print("\n--- Testing Predict: Critical Fall (No Motion) ---")
    payload = {
        "behavioral": { "acc": [15.0, 5.0, -2.0, 1.0, 1.0, 1.0, 15.0, 5.0, -2.0] },
        "environment": { "camera_motion": False }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Alert Level: {response.json().get('alert_level')}")
        print(f"Predictions: {response.json().get('predictions')}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_health()
    test_predict_normal()
    test_predict_critical_fall()

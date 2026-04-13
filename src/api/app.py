
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# --- 1. Configuration & Model Loading ---
MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/'))

def load_models():
    models = {}
    try:
        # Pillar A: Health Risk LSTM (WESAD)
        models['risk_lstm'] = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'health_risk_v1.h5'))
        # Pillar A: Clinical Risk Classification (MIMIC)
        models['clinical_rf'] = joblib.load(os.path.join(MODELS_DIR, 'risk_classification_v1.joblib'))
        # Pillar B: Fall Detection (SisFall)
        models['fall_rf'] = joblib.load(os.path.join(MODELS_DIR, 'fall_detection_v1.joblib'))
        # Pillar B: Behavioral Anomaly Detector (HMOG)
        models['behavior_if'] = joblib.load(os.path.join(MODELS_DIR, 'behavioral_biomarker_v2.joblib'))
        print("✅ All health monitoring models loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Model loading partial/failed. {e}")
    return models

health_models = load_models()

# --- 2. Helper Functions ---
def determine_alert_level(results):
    """
    Unified logic to determine alert severity based on all pillars.
    Levels: Normal -> Warning -> Critical
    """
    scores = []
    
    # Critical triggers
    if results.get('fall_status') == "CONFIRMED FALL":
        return "CRITICAL"
    if results.get('clinical_risk') == 1: # Sepsis/Critical label from MIMIC
        return "CRITICAL"
    
    # Warning triggers
    if results.get('behavioral_anomaly') == -1: # Isolation Forest anomaly
        scores.append(1)
    if results.get('vitals_risk_score', 0) > 0.7: # High probability from LSTM
        scores.append(1)
        
    if len(scores) >= 2:
        return "CRITICAL"
    elif len(scores) == 1:
        return "Warning"
    
    return "Normal"

# --- 3. API Endpoints ---

@app.route('/predict', methods=['POST'])
def predict():
    try:
        payload = request.json
        # Expected Payload:
        # {
        #   "vitals": { "hr": 72, "sbp": 120, "dbp": 80, "temp": 36.6, "spo2": 98, "age": 70, ... },
        #   "vitals_series": [[ax, ay, az], ...], (100 samples for LSTM)
        #   "behavioral": { "acc": [x, y, z] },
        #   "behavioral_stats": [mean_x, mean_y, mean_z, std_x, std_y, std_z, mag_mean],
        #   "environment": { "camera_motion": bool }
        # }

        results = {}
        
        # --- Pillar A: Vitals (Clinical & Sequence) ---
        if 'vitals' in payload and 'clinical_rf' in health_models:
            v = payload['vitals']
            # Match the 13 features from Risk_Classification_Training
            clinical_features = np.array([[
                v.get('age', 0), v.get('hr', 70), v.get('sbp', 120), v.get('dbp', 80), 
                v.get('map', 90), v.get('temp', 37), v.get('spo2', 98), v.get('resp', 16),
                v.get('wbc', 10), v.get('lactate', 1), v.get('glucose', 100), 
                v.get('sofa', 0), v.get('qsofa', 0)
            ]])
            results['clinical_risk'] = int(health_models['clinical_rf'].predict(clinical_features)[0])

        if 'vitals_series' in payload and 'risk_lstm' in health_models:
            series = np.array(payload['vitals_series'])
            if series.shape == (100, 3):
                series = series.reshape(1, 100, 3)
                results['vitals_risk_score'] = float(health_models['risk_lstm'].predict(series)[0].max())

        # --- Pillar B: Behavioral (Fall & Anomaly) ---
        if 'behavioral' in payload and 'fall_rf' in health_models:
            acc = np.array(payload['behavioral']['acc']).reshape(1, -1)
            fall_detected = bool(health_models['fall_rf'].predict(acc)[0])
            
            if fall_detected:
                camera_sees_motion = payload.get('environment', {}).get('camera_motion', True)
                results['fall_status'] = "CONFIRMED FALL" if not camera_sees_motion else "POSSIBLE FALL"
            else:
                results['fall_status'] = "No Fall"

        if 'behavioral_stats' in payload and 'behavior_if' in health_models:
            stats = np.array(payload['behavioral_stats']).reshape(1, -1)
            results['behavioral_anomaly'] = int(health_models['behavior_if'].predict(stats)[0])

        # --- Final Assessment ---
        final_alert = determine_alert_level(results)

        return jsonify({
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "predictions": results,
            "alert_level": final_alert
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "models_status": {k: "loaded" for k in health_models.keys()}
    })

if __name__ == '__main__':
    # Use environment variable for port to support cloud deployment
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Elderly Health AI Production API starting on port {port}...")
    app.run(host='0.0.0.0', port=port)

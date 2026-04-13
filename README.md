# Elderly Health Monitoring AI System (老人健康监测AI系统)

## Project Overview
This project aims to develop a real-time health monitoring and risk prediction system for the elderly using wearable sensor data. The system integrates multiple datasets to train AI models for health risk assessment and fall detection.

## Core Features
1. **Real-time Data Collection:** Processes heart rate, movement, and activity data from wearable sensors.
2. **Health Risk Prediction (LSTM):** Predicts health risk levels (Low, Medium, High) based on time-series wearable data.
3. **Fall Detection:** Real-time detection of falls using accelerometer data.
4. **Risk Classification:** Comprehensive risk assessment based on medical history and real-time vitals.
5. **Real-time Alerts:** Notifies users and family members of critical health events.

## Integrated Datasets
- **WESAD:** Wearable Stress and Affect Detection dataset (Heart rate, movement).
- **MIMIC-IV:** Medical Information Mart for Intensive Care IV (Clinical vitals and patient outcomes).
- **SisFall:** A Fall and Movement Dataset for Accelerometer-based Studies (Fall detection).

## Project Structure
- `data/`: Raw and processed datasets.
- `models/`: Trained AI models and model definitions.
- `notebooks/`: Data analysis and model experimentation.
- `src/`: Core source code for preprocessing, models, and API.
  - `src/data/`: Data loading and preprocessing logic.
  - `src/models/`: AI model definitions (LSTM, Fall detection).
  - `src/api/`: Flask API for real-time monitoring.
  - `src/utils/`: Helper utilities and logging.

## Workflow
1. **Data Preprocessing:** Clean and normalize wearable data into time-series sequences.
2. **Model Processing:** Use LSTM for risk prediction and classifiers for fall detection.
3. **API Response:** Provide endpoints for real-time risk assessment and alerts.

## Requirements
- TensorFlow / Keras
- Pandas / NumPy / Scikit-Learn
- Flask
- Google Cloud AI Platform / Storage / Logging

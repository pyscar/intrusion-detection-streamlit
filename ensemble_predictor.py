import joblib
import json
import os
import numpy as np

# Load models and preprocessors
models = {}

def load_model(domain):
    if domain not in models:
        path = f"models/{domain}"
        if domain == 'hospital':
            from tensorflow.keras.models import load_model
            model = load_model(f"{path}/hospital_cnn_lstm_model.keras")
            scaler = joblib.load(f"{path}/hospital_intrusion_scaler.pkl")
        else:
            model = joblib.load(f"{path}/model.pkl")
            scaler = joblib.load(f"{path}/scaler.pkl")
        models[domain] = (model, scaler)
    return models[domain]

def predict_intrusion(domain, input_data):
    model, scaler = load_model(domain)
    X = np.array(input_data).reshape(1, -1)
    X_scaled = scaler.transform(X)

    if domain == 'hospital':
        X_scaled = np.expand_dims(X_scaled, axis=1)
        prob = model.predict(X_scaled)[0][0]
        pred = int(prob > 0.5)
    else:
        pred = model.predict(X_scaled)[0]
    
    return {"prediction": int(pred)}

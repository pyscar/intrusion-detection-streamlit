import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model
import sys
import streamlit as st



st.set_page_config(page_title="Intrusion Detection Dashboard", layout="wide")
st.title("🚨 Ensemble Intrusion Detection System")
st.markdown("Select a model and input sample features to classify network traffic as Normal or Attack.")

# --- Sidebar to choose the model ---
domain = st.sidebar.selectbox("Select Domain", ["can", "iot_logistic", "hospital", "nsl_kdd"])

# --- Load model and metadata ---
model = None
scaler = None
feature_cols = []
label_encoder = None

model_path = f"models/{domain}"

if domain == "hospital":
    model = load_model(f"{model_path}/hospital_cnn_lstm_model.keras")
    scaler = joblib.load(f"{model_path}/hospital_intrusion_scaler.pkl")
    feature_cols = [
        'frame.time_delta', 'frame.time_relative', 'frame.len', 'tcp.srcport',
        'tcp.dstport', 'tcp.flags', 'tcp.time_delta', 'tcp.len', 'tcp.ack',
        'tcp.connection.fin', 'tcp.connection.rst', 'tcp.connection.sack',
        'tcp.connection.syn', 'tcp.flags.ack', 'tcp.flags.fin', 'tcp.flags.push',
        'tcp.flags.reset', 'tcp.flags.syn', 'tcp.flags.urg', 'tcp.hdr_len',
        'tcp.pdu.size', 'tcp.window_size_value', 'mqtt.clientid_len',
        'mqtt.conack.flags', 'mqtt.conack.val', 'mqtt.conflag.passwd',
        'mqtt.conflag.qos', 'mqtt.conflag.reserved', 'mqtt.conflag.retain',
        'mqtt.conflag.willflag', 'mqtt.conflags', 'mqtt.dupflag', 'mqtt.hdrflags',
        'mqtt.kalive', 'mqtt.len', 'mqtt.msgtype', 'mqtt.qos', 'mqtt.retain',
        'mqtt.topic_len', 'mqtt.ver', 'mqtt.willmsg_len', 'ip.proto', 'ip.ttl'
    ]
else:
    model = joblib.load(f"{model_path}/model.pkl")
    scaler = joblib.load(f"{model_path}/scaler.pkl")
    with open(f"{model_path}/metadata.json") as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    if "classes" in meta:
        label_encoder = joblib.load(f"{model_path}/label_encoder.pkl")

    if domain == "nsl_kdd":
        base_values = [
            0, 181, 5450, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            511, 511, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00,
            255, 255, 1.00, 0.00, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00,
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]
        if len(base_values) < len(feature_cols):
            sample_inputs_nsl = base_values + [0] * (len(feature_cols) - len(base_values))
        else:
            sample_inputs_nsl = base_values[:len(feature_cols)]

# --- Collect input ---
st.subheader(f"🧪 Input Features for: `{domain.upper()}`")

sample_inputs = {
    "hospital": [
        0.002, 15.24, 60, 1883, 443, 24, 0.004, 52, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 20,
        100, 65535, 12, 1, 0, 0, 0, 0, 1, 1, 0, 0, 32, 60, 3, 8, 1, 0, 10, 4, 0, 6, 64
    ],
    "can": [
        123456.78, 8, 255, 0, 128, 255, 0, 64, 128, 255, 0,
        127.5, 75.2, 1020, 255, 0, 255, 0.85, 1.2, 13,
        1020, 127.5, 75.2, 255, 0, 255, 127.5, 7
    ],
    "iot_logistic": [
        0.5, 1.2, 0.0, 0.75, 0.8, 0.9, 0.4, 0.3, 1.0, 0.0,
        0.1, 0.2, 0.3, 0.1, 0.6, 0.3, 0.7, 0.4, 0.6, 0.5,
        0.2, 0.8, 0.9, 0.4, 0.3, 0.1, 0.2, 0.6, 0.3, 0.7,
        0.5, 0.9, 0.2, 0.6, 0.8, 0.1, 0.3, 0.7, 0.6, 0.5,
        0.2, 0.1, 0.9, 0.8, 0.7, 0.6
    ],
    "nsl_kdd": sample_inputs_nsl if domain == "nsl_kdd" else []
}

inputs = []
input_method = st.radio("Choose input method:", ["Manual Input", "Use Sample Input", "Paste Comma-Separated Input", "Upload CSV"])

if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("📁 Upload a CSV file with feature values", type="csv")
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        if list(input_df.columns) != feature_cols:
            st.error("❌ Column mismatch. Please upload a file with correct feature columns.")
            st.stop()
        inputs = input_df.iloc[0].tolist()
elif input_method == "Use Sample Input":
    if domain in sample_inputs:
        inputs = sample_inputs[domain]
        st.success("✅ Sample input loaded.")
    else:
        st.warning("⚠️ No sample input defined for this domain.")
elif input_method == "Paste Comma-Separated Input":
    raw_input = st.text_area("Paste comma-separated values matching the required feature order:")
    if raw_input:
        try:
            values = [float(x.strip()) for x in raw_input.split(",")]
            if len(values) != len(feature_cols):
                st.error(f"❌ Expected {len(feature_cols)} values, but got {len(values)}")
            else:
                inputs = values
                st.success("✅ Parsed input successfully.")
        except:
            st.error("❌ Invalid input. Ensure all values are numeric and comma-separated.")
else:
    cols = st.columns(2)
    for idx, col in enumerate(feature_cols):
        default = 0.0 if "int" in col or "len" in col or "port" in col else 0.1
        val = cols[idx % 2].number_input(f"{col}", value=default, format="%f")
        inputs.append(val)

# --- Predict ---
if st.button("🔍 Predict") and inputs:
    try:
        input_df = pd.DataFrame([inputs], columns=feature_cols)
        scaled = scaler.transform(input_df)

        if domain == "hospital":
            reshaped = scaled.reshape((1, 1, scaled.shape[1]))
            pred = model.predict(reshaped)[0][0]
            label = "Attack" if pred > 0.5 else "Normal"
            confidence = float(pred)
        else:
            pred = model.predict(scaled)[0]
            if hasattr(model, 'predict_proba'):
                confidence = model.predict_proba(scaled)[0][int(pred)]
            else:
                confidence = 1.0
            label = pred
            if label_encoder:
                label = label_encoder.inverse_transform([pred])[0]

        st.success(f"✅ Prediction: **{label}**")
        st.metric("Confidence", f"{confidence:.4f}")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

#.venv\Scripts\activate

#streamlit run app.py


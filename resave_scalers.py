import joblib

scaler_paths = {
    "can": (
        r"D:\intrusion_project\models\can\scaler.pkl",
        r"D:\intrusion_project\models\can\scaler_resaved.pkl"
    ),
    "hospital": (
        r"D:\intrusion_project\models\hospital\hospital_intrusion_scaler.pkl",
        r"D:\intrusion_project\models\hospital\hospital_intrusion_scaler_resaved.pkl"
    ),
    "iot_logistic": (
        r"D:\intrusion_project\models\iot_logistic\scaler.pkl",
        r"D:\intrusion_project\models\iot_logistic\scaler_resaved.pkl"
    ),
    "nsl_kdd": (
        r"D:\intrusion_project\models\nsl_kdd\scaler.pkl",
        r"D:\intrusion_project\models\nsl_kdd\scaler_resaved.pkl"
    )
}

for name, (load_path, save_path) in scaler_paths.items():
    print(f"[🔄] Processing: {name}")
    try:
        scaler = joblib.load(load_path)
        joblib.dump(scaler, save_path)
        print(f"[✅] Resaved to: {save_path}\n")
    except Exception as e:
        print(f"[❌] Failed for {name}: {e}\n")

#python resave_scalers.py

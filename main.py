import pandas as pd
import numpy as np
import joblib
import os
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Evidently AI Imports
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.tests import TestShareOfDriftedColumns

# --- CONFIG ---
MODEL_PATH = "diabetes_model.pkl"
SCALER_PATH = "scaler.pkl"
REFERENCE_DATA_PATH = "reference_data.csv"
LOG_FILE = "production_logs.csv"
DRIFT_THRESHOLD = 0.3

app = FastAPI(title="Diabetes MLOps Pipeline")

# --- 1. ROBUST TRAINING ---
def train_model():
    if not os.path.exists('diabetes.csv'):
        print("❌ Error: diabetes.csv not found!")
        return
   
    df = pd.read_csv('diabetes.csv')
    cols_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_fix:
        df[col] = df[col].replace(0, df[col].median())
   
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
   
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
   
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
   
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    X_train.to_csv(REFERENCE_DATA_PATH, index=False)
    print("✅ Training complete. Artifacts saved.")

# --- 2. ALERTING LOGIC ---
def run_drift_check():
    if not os.path.exists(LOG_FILE) or not os.path.exists(REFERENCE_DATA_PATH):
        return
   
    try:
        ref_df = pd.read_csv(REFERENCE_DATA_PATH)
        cur_df = pd.read_csv(LOG_FILE)

        if len(cur_df) < 5: return

        drift_suite = TestSuite(tests=[TestShareOfDriftedColumns(lt=DRIFT_THRESHOLD)])
        drift_suite.run(reference_data=ref_df, current_data=cur_df.tail(50))
       
        result = drift_suite.as_dict()
        if not result["tests"][0]["parameters"]["condition"]["passed"]:
            print(f"\n🚨 ALERT: Data Drift detected ({result['tests'][0]['parameters']['real_value']:.2%})!")
    except Exception as e:
        print(f"⚠️ Drift check error: {e}")

# --- 3. API ENDPOINTS ---
class Patient(BaseModel):
    Pregnancies: int; Glucose: float; BloodPressure: float; SkinThickness: float
    Insulin: float; BMI: float; DiabetesPedigreeFunction: float; Age: int

@app.get("/", response_class=HTMLResponse)
async def home():
    return "<h1>API is running</h1><p>Send POST to /predict or <a href='/docs'>/docs</a> or visit <a href='/report'>/report</a></p>"

@app.post("/predict")
async def predict(data: Patient, background_tasks: BackgroundTasks):
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
       
        input_df = pd.DataFrame([data.dict()])
       
        # Log data for monitoring
        input_df.to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
       
        scaled = scaler.transform(input_df)
        pred = int(model.predict(scaled)[0])
       
        background_tasks.add_task(run_drift_check)
        return {"prediction": "Diabetic" if pred == 1 else "Healthy"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report")
async def get_report():
    # 1. Check if files exist
    if not os.path.exists(LOG_FILE) or not os.path.exists(REFERENCE_DATA_PATH):
        return HTMLResponse("<h1>No data logged yet</h1><p>Please make a few predictions first.</p>")

    try:
        ref_df = pd.read_csv(REFERENCE_DATA_PATH)
        cur_df = pd.read_csv(LOG_FILE)

        # 2. Match Columns (Crucial for Evidently)
        cur_df = cur_df[ref_df.columns]

        # 3. Generate Report
        report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
        report.run(reference_data=ref_df, current_data=cur_df)
       
        report.save_html("drift_report.html")
       
        with open("drift_report.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
       
    except Exception as e:
        print(f"DEBUG ERROR: {e}") # This will show in your terminal
        return HTMLResponse(f"<h1>Report Error</h1><p>{str(e)}</p>")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH): train_model()
    uvicorn.run(app, host="127.0.0.1", port=8000)

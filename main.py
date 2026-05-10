import os
import csv
import uuid
import random
import joblib
import pandas as pd
import uvicorn
from datetime import datetime
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Evidently AI
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

# --- CONFIG ---
REPORTS_DIR = "drift_reports"
LOG_FILE = "prediction_logs.csv"
MODEL_PATH = "diabetes_model.pkl"
REFERENCE_DATA = "reference_data.csv" # Created during initial training
MODEL_NAME = "Diabetes_Classifier_Model"  

os.makedirs(REPORTS_DIR, exist_ok=True)
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")


# --- 1. DEFINE THIS GLOBALLY AT THE TOP ---
field_info = {
    "Pregnancies": {"min": 0, "max": 20, "placeholder": "Count"},
    "Glucose": {"min": 0, "max": 200, "placeholder": "Glucose"},
    "BloodPressure": {"min": 0, "max": 140, "placeholder": "BP"},
    "SkinThickness": {"min": 0, "max": 100, "placeholder": "Skin mm"},
    "Insulin": {"min": 0, "max": 900, "placeholder": "Insulin"},
    "BMI": {"min": 0, "max": 70, "placeholder": "BMI"},
    "DiabetesPedigreeFunction": {"min": 0.0, "max": 3.0, "placeholder": "Pedigree"},
    "Age": {"min": 21, "max": 100, "placeholder": "Years"},
}

# --- UTILS ---

def save_prediction(input_dict, prediction):
    """Saves prediction to CSV for the history page."""
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Glucose", "BMI", "Age", "Result"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            input_dict.get("Glucose"),
            input_dict.get("BMI"),
            input_dict.get("Age"),
            "Diabetic" if prediction == 1 else "Healthy"
        ])

def run_drift_analysis(current_df):
    """Compares current input against reference data and saves HTML report."""
    if not os.path.exists(REFERENCE_DATA):
        return None
    
    ref_df = pd.read_csv(REFERENCE_DATA)
    # Ensure columns match
    current_df = current_df[ref_df.columns]
    
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=ref_df, current_data=current_df)
    
    report_filename = f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    report_path = os.path.join(REPORTS_DIR, report_filename)
    report.save_html(report_path)
    return report_filename

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    reports = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".html")]
    reports.sort(reverse=True)
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "field_info": field_info,  # <--- MUST ADD THIS
        "reports": reports,
        "prediction": None
    })

@app.post("/predict-ui", response_class=HTMLResponse)
async def predict_ui(request: Request):
    form = await request.form()
    input_data = {k: float(v) for k, v in form.items()}
    df = pd.DataFrame([input_data])
    
    model = joblib.load(MODEL_PATH)
    pred = int(model.predict(df)[0])
    
    save_prediction(input_data, pred)
    run_drift_analysis(df)
    
    reports = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".html")]
    reports.sort(reverse=True)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "field_info": field_info,  # <--- MUST ADD THIS TOO
        "prediction": "High Risk" if pred == 1 else "Low Risk",
        "reports": reports
    })

@app.post("/train", response_class=HTMLResponse)
async def train_model_ui(request: Request):
    try:
        # 1. Setup MLflow
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(MODEL_NAME)
        
        # 2. Prepare Data
        df = pd.read_csv("diabetes.csv")
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Random Hyperparameter
        n_est = random.randint(10, 200)
        
        with mlflow.start_run() as run:
            model = RandomForestClassifier(n_estimators=n_est, random_state=42)
            model.fit(X_train, y_train)
            
            acc = accuracy_score(y_test, model.predict(X_test))
            
            # Log to MLflow
            mlflow.log_param("n_estimators", n_est)
            mlflow.log_metric("accuracy", round(acc, 4))
            
            # Register Model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=MODEL_NAME
            )
            
            # Update local model file for immediate use in predictions
            joblib.dump(model, "diabetes_model.pkl")

        # 4. Refresh the home page with a success message
        reports = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".html")]
        reports.sort(reverse=True)
        
        client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")

        return templates.TemplateResponse("index.html", {
            "request": request,
            "field_info": field_info,
            "reports": reports,
            "model_versions": versions,
            "prediction": f"✅ New Model Trained! Estimators: {n_est}, Accuracy: {round(acc, 2)}",
            "status_color": "success"
        })
    except Exception as e:
        print(f"Training Error: {e}")
        return HTMLResponse(content=f"Training Failed: {e}", status_code=500)

@app.get("/prediction-history", response_class=HTMLResponse)
async def prediction_history(request: Request):
    """Reads the CSV and displays it in a table."""
    data = []
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        data = df.tail(20).to_dict(orient="records") # Show last 20
    
    return templates.TemplateResponse("history.html", {
        "request": request,
        "history": data
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
import os
import joblib
import pandas as pd
import uvicorn
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- CONFIG ---
MLFLOW_URI = "http://127.0.0.1:5000"
MODEL_NAME = "Diabetes_Classifier_Production"
API_PORT = 8000

app = FastAPI(title="FastAPI + MLflow")

# Connect to the server running in Terminal 1
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Diabetes_Inference_Monitoring")

# --- 1. MODEL REGISTRY LOGIC ---
def train_and_register():
    print("🧠 Training and Registering Model to MLflow...")
    
    # Load dataset
    df = pd.read_csv('diabetes.csv')
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[col] = df[col].replace(0, df[col].median())
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    with mlflow.start_run(run_name="Deployment_Run"):
        model = RandomForestClassifier(n_estimators=100, max_depth=5)
        model.fit(X_train_scaled, y_train)
        
        # Model Signature (Critical for the Registry UI)
        signature = infer_signature(X_train, model.predict(X_train_scaled))
        
        # Log and Register the model into the 'Models' tab
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME
        )
        
        # Save artifacts locally for the FastAPI service
        joblib.dump(model, "diabetes_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        X_train.to_csv("reference_data.csv", index=False)
        
    print(f"✅ Success! Model '{MODEL_NAME}' is now in the MLflow Registry.")

# --- 2. API ENDPOINTS ---
class Patient(BaseModel):
    Pregnancies: int; Glucose: float; BloodPressure: float; SkinThickness: float
    Insulin: float; BMI: float; DiabetesPedigreeFunction: float; Age: int

@app.get("/", response_class=HTMLResponse)
async def home():
    return f"""
    <body style="font-family:sans-serif; padding:40px;">
        <h1>MLOps Control Center</h1>
        <p>🟢 <b>API:</b> Running on port {API_PORT}</p>
        <p>📊 <b>MLflow:</b> <a href="{MLFLOW_URI}" target="_blank">Open Dashboard</a> (Check 'Models' tab)</p>
        <p>🛠️ <b>Docs:</b> <a href="/docs">Swagger UI</a></p>
    </body>
    """

@app.post("/predict")
async def predict(data: Patient):
    if not os.path.exists("diabetes_model.pkl"):
        raise HTTPException(status_code=500, detail="Model not trained yet.")
    
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    input_df = pd.DataFrame([data.dict()])
    scaled = scaler.transform(input_df)
    pred = int(model.predict(scaled)[0])
    return {"prediction": "Diabetic" if pred == 1 else "Healthy"}

# --- 3. RUNNER ---
if __name__ == "__main__":
    # Ensure model is registered before starting API
    if not os.path.exists("diabetes_model.pkl"):
        train_and_register()
    else:
        print("ℹ️ Local model found. Starting API...")

    uvicorn.run(app, host="127.0.0.1", port=API_PORT)
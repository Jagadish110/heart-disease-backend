from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import os
import numpy as np

app = FastAPI()

# Define request schema
class HeartData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_webpage.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError as e:
    raise RuntimeError(f"Required file not found: {e}")

@app.post("/predict")
def predict(data: HeartData):
    try:
        input_data = np.array([[data.age, data.sex, data.cp, data.trestbps, data.chol,
                                data.fbs, data.restecg, data.thalach, data.exang,
                                data.oldpeak, data.slope, data.ca, data.thal]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        return {"prediction": int(prediction)}  # 1 = Heart Disease, 0 = No Disease
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

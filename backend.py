from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import os

# Load the model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models\heart_webpage.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


app = FastAPI()

# Define the input data model
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

@app.post("/predict")
def predict(data: HeartData):
    features = np.array([[ 
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])
    
    scaled = MODEL_PATH.transform(features)
    prediction = model.predict(scaled)[0]
    result = " Heart Disease" if prediction == 1 else "No Heart Disease"
    return {"prediction": result}

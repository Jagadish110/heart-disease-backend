# backend.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Define input schema
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

# Load model
model = joblib.load("models/heart_webpage.pkl")  # Make sure model.pkl is present in root folder

@app.get("/")
def read_root():
    return {"message": "Model loaded successfully!"}

@app.post("/predict")
def predict(data: HeartData):
    input_array = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal
    ]])
    prediction = model.predict(input_array)
    return {"prediction": int(prediction[0])}

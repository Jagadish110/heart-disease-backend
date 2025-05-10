from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI()

# Configure CORS to allow requests from Streamlit app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your Streamlit URL after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input model for heart disease features
class PredictionInput(BaseModel):
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

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "heart_disease_model.pkl")
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception(f"Model file not found at {MODEL_PATH}")

# Prediction endpoint
@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        # Convert input to numpy array for prediction
        input_data = np.array([[
            data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
            data.restecg, data.thalach, data.exang, data.oldpeak, data.slope,
            data.ca, data.thal
        ]])

        # Make prediction
        prediction = model.predict(input_data)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}
import os
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Dynamic model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_webpage.pkl")

# Debug: Print the model path
print(f"Attempting to load model from: {MODEL_PATH}")

# Load model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except FileNotFoundError as e:
    print(f"Error: Model file not found at {MODEL_PATH}")
    raise Exception(f"Model file not found at {MODEL_PATH}")

@app.get("/")
async def root():
    return {"message": "Model loaded successfully!"}

# Pydantic model for incoming request data
class HeartDiseaseRequest(BaseModel):
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
async def predict(data: HeartDiseaseRequest):
    # Convert incoming Pydantic model to a list of features
    features = [
        data.age, data.sex, data.cp, data.trestbps,
        data.chol, data.fbs, data.restecg, data.thalach,
        data.exang, data.oldpeak, data.slope, data.ca, data.thal
    ]
    
    try:
        # Predict using the model
        prediction = model.predict([features])  # Assuming the model expects a 2D array
        result = {"prediction": prediction[0]}  # Return the first prediction
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os
import boto3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS to allow requests from Streamlit app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your Streamlit URL after deployment, e.g., ["https://heart-disease-frontend.onrender.com"]
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

# Load the trained model and scaler
MODEL_PATH = "/tmp/heart_disease_model.pkl" if os.getenv("AWS_S3_BUCKET") else os.path.join(os.path.dirname(__file__), "models", "heart_disease_model.pkl")
SCALER_PATH = "/tmp/scaler.pkl" if os.getenv("AWS_S3_BUCKET") else os.path.join(os.path.dirname(__file__), "models", "scaler.pkl")

try:
    if os.getenv("AWS_S3_BUCKET"):
        logger.info("Downloading model and scaler from S3...")
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY")
        )
        s3.download_file(os.getenv("AWS_S3_BUCKET"), "heart_disease_model.pkl", MODEL_PATH)
        s3.download_file(os.getenv("AWS_S3_BUCKET"), "scaler.pkl", SCALER_PATH)
    
    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    # Load scaler
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError as e:
    logger.error(f"Model or scaler file not found: {e}")
    raise Exception(f"Model or scaler file not found: {e}")
except Exception as e:
    logger.error(f"Error loading model or scaler: {str(e)}")
    raise Exception(f"Error loading model or scaler: {str(e)}")

# Prediction endpoint
@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        # Convert input to numpy array
        input_data = np.array([[
            data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
            data.restecg, data.thalach, data.exang, data.oldpeak, data.slope,
            data.ca, data.thal
        ]])
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        # Make prediction
        prediction = model.predict(input_data_scaled)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "healthy"}
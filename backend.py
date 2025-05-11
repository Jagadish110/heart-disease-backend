import os
import pickle
from fastapi import FastAPI, HTTPException
from asgiref.wsgi import WsgiToAsgi

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

# Wrap FastAPI app for WSGI servers (e.g., waitress on Windows)
wsgi_app = WsgiToAsgi(app)
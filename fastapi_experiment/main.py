# ==============================
# Step 1: Import Libraries
# ==============================

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# ==============================
# Step 2: Train ML Model
# ==============================

# Load dataset
data = load_iris()

# Take only first 100 samples (Binary classification)
X = data.data[:100]
y = data.target[:100]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# ==============================
# Step 3: Create FastAPI App
# ==============================

app = FastAPI(
    title="Logistic Regression API",
    description="FastAPI model serving example",
    version="1.0"
)


# ==============================
# Step 4: Define Input Schema
# ==============================

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float


# ==============================
# Step 5: Create Prediction Endpoint
# ==============================

@app.post("/predict")
def predict(data: InputData):

    # Convert input to numpy array
    features = np.array([
        [
            data.feature1,
            data.feature2,
            data.feature3,
            data.feature4
        ]
    ])

    # Make prediction
    prediction = model.predict(features)

    return {
        "prediction": int(prediction[0])
    }


# ==============================
# Step 6: Root Endpoint
# ==============================

@app.get("/")
def root():
    return {
        "message": "Logistic Regression API",
        "endpoints": {
            "/health": "GET - Health check",
            "/predict": "POST - Make prediction",
            "/predict-sample": "GET - Test prediction with sample data",
            "/docs": "GET - Interactive API documentation"
        }
    }


# ==============================
# Step 7: Health Check Endpoint
# ==============================

@app.get("/health")
def health_check():
    return {"status": "API is running successfully"}


# ==============================
# Step 8: Sample Prediction Endpoint (GET)
# ==============================

@app.get("/predict-sample")
def predict_sample():
    # Sample iris data
    sample_features = np.array([[5.1, 3.5, 1.4, 0.2]])
    
    prediction = model.predict(sample_features)
    
    return {
        "input": {
            "feature1": 5.1,
            "feature2": 3.5,
            "feature3": 1.4,
            "feature4": 0.2
        },
        "prediction": int(prediction[0]),
        "note": "Use POST /predict for custom predictions"
    }
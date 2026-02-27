from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load saved model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.feature1,
                          data.feature2,
                          data.feature3,
                          data.feature4]])

    prediction = model.predict(features)

    return {"prediction": int(prediction[0])}
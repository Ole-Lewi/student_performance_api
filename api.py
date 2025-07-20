from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

#Load the trained model
model = joblib.load('Linear Regression.pkl')

#input format
class StudyHours(BaseModel):
    hours: float

# Create FastAPI app   
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Student Performance Predictor API"}

@app.post("/predict")
def predict_score(study_hours: StudyHours):
    hours_studied = np.array([[study_hours.hours]])
    predicted_score = model.predict(hours_studied)
    return {
        "studied_hours": study_hours.hours,
        "predicted_score": predicted_score[0]
        }
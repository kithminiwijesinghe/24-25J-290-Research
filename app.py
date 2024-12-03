from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from typing import Literal

# Initialize FastAPI app
app = FastAPI(
    title="BrainBounce API",
    description="API for predicting Progression Status ('Promoted' or 'Repeat') based on user performance metrics.",
    version="1.0.0"
)



# Load the trained model
model = joblib.load("../../models/F2T4/f2t4.pkl")

# Input schema for validation
class PredictionInput(BaseModel):
    Attempts_Made: int = Field(..., gt=0, le=5, description="Number of attempts made (1-5)")
    Accuracy: float = Field(..., ge=0, le=100, description="Accuracy percentage (0-100)")
    Pronunciation_Score: float = Field(..., ge=0, le=100, description="Pronunciation score (0-100)")
    Time_Taken_s: int = Field(..., gt=0, le=600, description="Time taken in seconds (1-600)")
    Hints_Used: int = Field(..., ge=0, le=5, description="Number of hints used (0-5)")
    Reward_Points: int = Field(..., ge=0, le=100, description="Reward points earned (0-100)")

@app.post("/predict")
async def predict(input_data: PredictionInput):
    """
    Predicts the progression status ('Promoted' or 'Repeat') based on user performance metrics.
    """
    try:
        # Convert input data to DataFrame and rename columns to match the training feature names
        input_df = pd.DataFrame([input_data.dict()])
        input_df.columns = [
            "Attempts Made", "Accuracy", "Pronunciation Score", "Time Taken (s)", "Hints Used", "Reward Points"
        ]

        # Make prediction
        prediction = model.predict(input_df)
        predicted_status = "Promoted" if prediction[0] == 1 else "Repeat"
        
        # Return prediction result
        return {"status": "success", "prediction": predicted_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


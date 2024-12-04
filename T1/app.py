import warnings

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved model and scalers
rf_model = joblib.load("../../models/PTDA/ptda.pkl")
scaler_X = joblib.load("../../data/PTDA/scaler_X.pkl")
scaler_y = joblib.load("../../data/PTDA/scaler_y.pkl")

app = FastAPI()

# Define the input data structure using Pydantic
class InputData(BaseModel):
    success_count: int
    attempt_count: int
    game_score_xp: int
    game_level: int
    engagement_time_mins: int

# Prediction function
def predict_improvement_score(success_count, attempt_count, game_score_xp, game_level, engagement_time_mins):
    user_inputs = np.array([success_count, attempt_count, game_score_xp, game_level, engagement_time_mins]).reshape(1, -1)

    warnings.filterwarnings("ignore")

    # Scale the user inputs
    user_inputs_scaled = scaler_X.transform(user_inputs)

    warnings.filterwarnings("default")

    # Make prediction
    predicted_score_scaled = rf_model.predict(user_inputs_scaled)

    predicted_score = scaler_y.inverse_transform(predicted_score_scaled.reshape(1, -1))

    return predicted_score[0][0] * 10  # Return the score multiplied by 10 for percentage

# Create a route for predictions
@app.post("/predict/")
def get_prediction(data: InputData):
    result = predict_improvement_score(
        data.success_count,
        data.attempt_count,
        data.game_score_xp,
        data.game_level,
        data.engagement_time_mins
    )
    return {"predicted_improvement_score": f"{result:.2f}%"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
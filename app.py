from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import pickle

app = FastAPI()

with open('../../models/Progress_Tracking/best_rtp_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

class PredictRequest(BaseModel):
    Time: float
    Score: float
    Attempts: int
    Level: int

@app.get('/')
async def read_root():
    return {"message": "Welcome to the Brain Bounce Progress Tracking API"}

@app.post('/predict')
async def predict(request: PredictRequest):
    X_new = [[request.Time, request.Score, request.Attempts, request.Level]]
    prediction = loaded_model.predict(X_new)[0]
    return {'Performance': prediction}

# main method
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
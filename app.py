from fastapi import FastAPI, Query
from datetime import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from typing import Optional
import os

app = FastAPI()

# Define the model path
model_path = '../models/T3/f4t3.keras'

# Check if the model file exists
if not os.path.exists(model_path):
    raise ValueError(f"File not found: filepath={model_path}. Please ensure the file is an accessible `.keras` zip file.")

# Load the saved model
model = tf.keras.models.load_model(model_path)

# Load the dataset and preprocessing steps
data = pd.read_csv('../data/T3/f4t3_data.csv')

le_main_level = LabelEncoder()
data['Main Level'] = le_main_level.fit_transform(data['Main Level'])

scaler = MinMaxScaler()
numerical_features = ['Engaging Time', 'Score', 'Attempts', 'Count of Attempts']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by=['ID', 'Date'])

# FastAPI endpoint
@app.get("/predict/")
async def predict_level(
    child_id: str = Query(..., description="Child ID (e.g., 'C001')"),
    date_str: str = Query(..., description="Date in YYYY-MM-DD format (future date)"),
):
    try:
        # 1. Convert date string to datetime object
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        today = datetime.now()

        # 2. Check if the input date is a future date
        if date_obj <= today:
            return {"error": "Please enter a future date for prediction."}

        # 3. Check if child_id is in the DataFrame
        if child_id not in data['ID'].unique():
            return {"error": f"No data found for child ID: {child_id}"}

        # 4. Filter data for the given child_id
        child_data = data[data['ID'] == child_id].copy()

        # 5. Get the latest 5 records (or less if not available)
        latest_records = child_data.tail(5)

        # 6. Check if we have enough data (at least 1 record)
        if len(latest_records) < 1:
            return {"error": "Not enough data for the given child."}

        # 7. Prepare input sequence
        input_sequence = latest_records[['Engaging Time', 'Score', 'Attempts', 'Count of Attempts']].values
        input_sequence = input_sequence.reshape(1, input_sequence.shape[0], input_sequence.shape[1])

        # 8. Make prediction
        prediction = model.predict(input_sequence)
        predicted_main_level = le_main_level.inverse_transform(np.round(prediction[0, 0]).astype(int).reshape(-1))[0]  # Get the single value
        predicted_sublevel = round(prediction[0, 1])

        # 9. Return prediction
        return {
            "child_id": child_id,
            "date": date_str,
            "predicted_main_level": predicted_main_level,
            "predicted_sublevel": predicted_sublevel,
        }

    except Exception as e:
        return {"error": str(e)}
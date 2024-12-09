from datetime import datetime, timedelta
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the saved model
model = keras.models.load_model('/content/drive/MyDrive/BrainBounce/F4T3/f4t3.keras')

# Load the dataset 
data = pd.read_csv('/content/drive/MyDrive/BrainBounce/F4T3/f4t3_data.csv')

le_main_level = LabelEncoder()
data['Main Level'] = le_main_level.fit_transform(data['Main Level'])

scaler = MinMaxScaler()
numerical_features = ['Engaging Time', 'Score', 'Attempts', 'Count of Attempts']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by=['ID', 'Date'])

def predict_level_for_date(child_id, date_str):
    # 1. Convert date string to datetime object
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    today = datetime.now()

    # 2. Check if the input date is a future date
    if date_obj <= today:
        print("Please enter a future date for prediction.")
        return

    # 3. Check if child_id is in the DataFrame
    if child_id not in data['ID'].unique():
        print(f"No data found for child ID: {child_id}")
        return

    # 4. Filter data for the given child_id
    child_data = data[data['ID'] == child_id].copy()

    # 5. Get the latest 5 records (or less if not available)
    latest_records = child_data.tail(5)  

    # 6. Check if we have enough data (at least 1 record)
    if len(latest_records) < 1:  
        print("Not enough data for the given child.")
        return

    # 7. Prepare input sequence
    input_sequence = latest_records[['Engaging Time', 'Score', 'Attempts', 'Count of Attempts']].values
    input_sequence = input_sequence.reshape(1, input_sequence.shape[0], input_sequence.shape[1])

    # 8. Make prediction
    prediction = model.predict(input_sequence)
    predicted_main_level = le_main_level.inverse_transform(np.round(prediction[0, 0]).astype(int).reshape(-1)) # Reshape to 1D array
    predicted_sublevel = round(prediction[0, 1])

    # 9. Print prediction
    print(f"Predicted Level for Child {child_id} on {date_str}:")
    print(f"Main Level: {predicted_main_level[0]}")
    print(f"Sublevel: {predicted_sublevel}")

# Example usage:
predict_level_for_date('C001', '2024-11-30')
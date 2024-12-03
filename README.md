Function 2: Supportive learning platform for children who have dyslexia disorder

Model 1: Speech Recognition and Pronunciation Scoring Model
     Use Technologies and Model: OpenAI's whisper, Transformers, PyTorch
     
     Model Inputs: String (Original text), File (Audio file .wav, .mp3)
     
     Model Output: Strings (Original text, Predicted text and Wer score(Word error rate))
     
     Model (GitHub or Drive URL): Model File
     
     Accuracy: pre-trained model

Model 4: Classification Model for Progress Tracking

     Use Technologies and Model: Scikit-learn, Decision Tree Classifier
     
     Model Inputs: Integer (Attempts_Made, Accuracy, Pronunciation_Score, Time_Taken, Hints_Used, Reward_Points)
     
     Model Output: Strings ("prediction": "Promoted" or "Repeat")
     
     Dataset (Drive or GitHub URL): https://drive.google.com/drive/folders/1O0I2B3OzoRwJ3xjyVNwuxQwDlA9ON5sz
     
     Model (GitHub or Drive URL): https://drive.google.com/drive/folders/1ppoVME2DuKY9OQtHieLw82G0v4VJNPgy
     
     How to Load and Get Prediction for One Input:
     
         import joblib
import pandas as pd

# Load the saved model
model = joblib.load("f2t4.pkl")

# Define a single input
test_input = {
    "Attempts Made": 2,         
    "Accuracy": 85.0,            
    "Pronunciation Score": 90.0, 
    "Time Taken (s)": 200,       
    "Hints Used": 1,             
    "Reward Points": 75          
}

# Convert the input into a DataFrame
test_df = pd.DataFrame([test_input])

# Make a prediction
prediction = model.predict(test_df)
predicted_status = "Promoted" if prediction[0] == 1 else "Repeat"

# Output the result
print(f"Predicted Progression Status: {predicted_status}")

API (TBD)
Use Technology: Flask, Swagger


Prerequisites

## Environment

1. Navigate to Your Project Directory `cd Function_2`
2. Create a Virtual Environment `python -m venv venv`
3. Activate Virtual Environment `source venv/Scripts/activate`
4. Deactivate Virtual Environment `deactivate` (Optional)

## Installation

1. Install Required Libraries `pip install -r requirements.txt`
2. Create requirements.txt `pip freeze > requirements.txt` (If needed)

## Run

Run application using `fastapi dev app.py` (default) command.

Task 01: `fastapi dev T1/app.py`
Task 02: `fastapi dev T2/app.py`
Task 03: `fastapi dev T3/app.py`
Task 04: `fastapi dev T4/app.py`

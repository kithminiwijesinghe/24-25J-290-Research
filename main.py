import joblib
import pandas as pd

# Load the saved model
model = joblib.load("/content/drive/MyDrive/BrainBounce/F2T4/f2t4.pkl")

# Define a single input (update the values as per your test case)
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
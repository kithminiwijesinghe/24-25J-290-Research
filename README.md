# 24-25J-290-Research

Brain-Bounce

Project Brain Bounce aims to develop an intelligent mobile tutoring system for children with various learning disorders including dysgraphia, dyslexia, short-term memory loss, and dyscalculia.
The system utilizes advanced technologies to provide personalized learning experiences, addressing specific challenges in writing, reading, memory retention, and mathematical understanding.

### Function 2: Supportive learning platform for children who have dyslexia disorder 
### IT21181610 Owner - K.S.K Wijesinghe Email: it21181610@my.sliit.lk
#### Model 1,2 and 3: Speech Recognition and Pronunciation Scoring Model

- **Use Technologies and Model**: OpenAI's whisper, Transformers, PyTorch
- **Model Inputs**: String (Original text), File (Audio file .wav, .mp3)
- **Model Output**: Strings (Original text, Predicted text)
- **Accuracy**: pre-trained model
---
#### Model 4: Classification Model for Progress Tracking

- **Use Technologies and Model**: Scikit-learn, Decision Tree Classifier
- **Model Inputs**: Integer (Attempts_Made, Accuracy, Pronunciation_Score, Time_Taken, Hints_Used, Reward_Points)
- **Model Output**: Strings ("prediction": "Promoted" or "Repeat")
- **Accuracy**: 0.98
- **How to Load and Get Prediction for One Input**:
    ```python
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
    ```

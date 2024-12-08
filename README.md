# 24-25J-290-Research

Brain-Bounce

Project Brain Bounce aims to develop an intelligent mobile tutoring system for children with various learning disorders including dysgraphia, dyslexia, short-term memory loss, and dyscalculia.
The system utilizes advanced technologies to provide personalized learning experiences, addressing specific challenges in writing, reading, memory retention, and mathematical understanding.
## 1. Functions

### Function 1: Supportive learning platform for children who have dysgraphia disorder
### IT21163272 Owner Yasathri G.A Email :it21163272@my.sliit.lk
#### Model 1: Convolutional Neural Network (CNN) for Handwriting Recognition

- *Use Technologies and Model*: Tensorflow, CNN
- *Model Inputs*: File (image)
- *Model Outputs*: String (Character or Number)
- *Accuracy*:0.92
- *How to Load and Get Prediction for One Input*:
    python
    from tensorflow.keras.models import load_model
    import numpy as np
    import cv2

    # Load the saved model

    model = load_model('Handwriting_recognition_1.keras')

    # Load the classes

    with open('classes3.txt', 'r') as f:
    classes = f.read().splitlines()
    def preprocess_image(image_path):
        img = cv2.imread(image_path)
        img = to_binary(img)
        img = img / 255.0  # Normalize
        img = img.reshape(1, 32, 32, 1)  # Reshape to match model input
        return img

    # Path to the new image

    image_path = '/content/dataset/Validation/Y/100.jpg'

    # Preprocess the image
    processed_image = preprocess_image(image_path)
    # Get the prediction
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = classes[predicted_class_index]

    print(f"Predicted Class: {predicted_class_label}")
    

#### Model 4: Classification Model for Progress Tracking

- *Use Technologies and Model*: Scikit-learn, Random Forest Regressor
- *Model Label*: Performance
- *Model Features*: Time, Score, Attempts, Level (int)
- *Accuracy*:0.95
- *How to Load and Get Prediction for One Input*:
    python
    import pickle

    with open('best_rtp_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    def predict():
        data = request.get_json()
        X_new = [data['Time'], data['Score'], data['Attempts'], data['Level']]
        prediction = model.predict([X_new])[0]
        return jsonify({'Performance': prediction})
    
---
---
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
### Function 3: Supportive learning platform for child who have short term memory loss disorder
### IT21159008 Owner - Hewage L.H.C.S. Email: it21159008@my.sliit.lk
#### Model 1: Sequence Prediction Model

- Use Technologies and Model: Scikit-learn, Logistic Regression
- Model Inputs: integers and strings
- Model Outputs: String (predicted_next_action, predicted_difficulty_level)
- Accuracy:0.99
- How to Load and Get Prediction for One Input:

# Load the saved model
    model_filename = ''
    loaded_model = joblib.load(model_filename)

    preprocessor_filename = ''
    loaded_preprocessor = joblib.load(preprocessor_filename)

    label_encoder_next_action_filename = ''
    label_encoder_next_action = joblib.load(label_encoder_next_action_filename)

    label_encoder_diff_level_filename = ''
    label_encoder_diff_level = joblib.load(label_encoder_diff_level_filename)

    # Define the structure of input data
    random_data = {
        'difficulty_level': [3],
        'number_attempts': [2],
        'time_spent': [30],
        'hints_used': [1],
        'is_correct': [1],
        'topic': ['algebra'],
        'subtopic': ['equations'],
        'age': [6], 
        'memory_recall_score': [75], 
        'engagement_level': [4], 
        'level_progression': [2],
        'attempts': [10],
        'game_type': ['puzzle'],
        'feedback_received': ['positive'], 
        'learning_preferences': ['visual'],
        'correct_answers': [8], 
        'response_time': [20],  
        'incorrect_answers': [2], 
        'completion_status': ['completed']
    }

    random_df = pd.DataFrame(random_data)

    # Preprocessing
    X_processed_random = loaded_preprocessor.transform(random_df)

    # Make Predictions
    predictions = loaded_model.predict(X_processed_random)

    # Decode Predictions (using your loaded LabelEncoders)
    predicted_next_action = label_encoder_next_action.inverse_transform([predictions[0][0]])[0]
    predicted_difficulty_level = label_encoder_diff_level.inverse_transform([predictions[0][1]])[0]

    # Print Predictions
    print("Predicted Next Action:", predicted_next_action)
    print("Predicted Difficulty Level:", predicted_difficulty_level)


  #### Model 3: Student Performance Recommendation Model

- Use Technologies and Model: Scikit-learn, Decision Tree Classifier
- Model Inputs: integer (time_taken, success_rate, errors, hints_used, completion_status, game level [sequence_master, shape_shifter or story_builder])
- Model Outputs: String (predicted next level: Advance or Repeat)
- Accuracy:0.99
- How to Load and Get Prediction for One Input:

# Load the model
    model = joblib.load('spr.pkl')
    le = joblib.load('spr_label_encoder.pkl')

    # Predict for a new input
    random_input = pd.DataFrame({
        'Time Taken (seconds)': [500],
        'Success Rate (%)': [85],
        'Errors': [1],
        'Hints Used': [2],
        'Completion Status': [1],
        'GameLevel_Sequence Master': [1],
        'GameLevel_Shape Shifter': [0],
        'GameLevel_Story Builder': [0]
    })

    prediction = model.predict(random_input)
    predicted_class = le.inverse_transform(prediction)
    print("Predicted Next Level:", predicted_class[0])

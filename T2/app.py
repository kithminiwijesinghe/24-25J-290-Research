from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.utils import img_to_array
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import uvicorn
import io

app = FastAPI()

# Load the trained model and class labels
with open('../../models/F4T2/class_labels.json', 'r') as f:
    class_labels = json.load(f)

model = tf.keras.models.load_model('../../models/F4T2/model.keras')

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):

    # Read the image file
    contents = await file.read()
    
    # Convert the image to a NumPy array
    img = Image.open(io.BytesIO(contents))
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]

    return {"predicted_class": predicted_class_label}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Run the FastAPI app